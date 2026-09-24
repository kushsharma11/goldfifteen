"""Provider contracts tested with explicit transport fixtures, never live secrets."""

import asyncio
import base64
import json
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from gold_model.config import Settings
from gold_model.providers import CSVComexProvider, DatabentoComexProvider, KalshiProvider, ProviderError, PythProvider
from gold_model.providers.base import utc

FEED = "a" * 64
START = datetime(2025, 1, 2, 14, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def isolated_provider_environment(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def settings(**kwargs):
    return Settings(_env_file=None, provider_min_interval_seconds=0, request_retries=0, **kwargs)


def pyth_payload(timestamp=START, price="430012345678", feed=FEED):
    return {"parsed": [{"id": feed, "price": {"price": price, "conf": "10000000", "expo": -8, "publish_time": int(timestamp.timestamp())}, "metadata": {"slot": 1}}]}


def market_payload(**kwargs):
    return {"ticker": "KXGOLD15M-TEST-00", "event_ticker": "KXGOLD15M-TEST", "open_time": START.isoformat(), "close_time": (START + timedelta(minutes=15)).isoformat(), "floor_strike": 4300.1, "strike_type": "greater_or_equal", "status": "settled", "result": "yes", "expiration_value": "4301.25", "settlement_value_dollars": "1.0000", "settlement_ts": (START + timedelta(minutes=16)).isoformat(), **kwargs}


def client_for(payload, status=200):
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(status, json=payload)))


def test_pyth_price_confidence_and_receive_time():
    raw = []

    async def run():
        async with client_for(pyth_payload()) as client:
            provider = PythProvider(settings(pyth_api_key="test", pyth_feed_id=FEED), client, raw_sink=lambda *args: raw.append(args))
            point = await provider.latest_price()
            assert point.price == pytest.approx(4300.12345678)
            assert point.confidence == 0.1
            assert point.timestamp == START
            assert point.available_at > START
            assert raw[0][2] == pyth_payload()
    asyncio.run(run())


def test_raw_is_persisted_before_pyth_normalization_failure():
    raw = []

    async def run():
        async with client_for(pyth_payload(price="-1")) as client:
            provider = PythProvider(settings(pyth_api_key="test", pyth_feed_id=FEED), client, raw_sink=lambda *args: raw.append(args))
            with pytest.raises(ProviderError, match="invalid price"):
                await provider.latest_price()
            assert len(raw) == 1
    asyncio.run(run())


def test_pyth_history_never_backdates_future_returned_tick():
    async def run():
        requested = []

        def response(request):
            requested.append(request.url.path)
            return httpx.Response(200, json=pyth_payload(START + timedelta(seconds=2)))

        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            provider = PythProvider(settings(pyth_api_key="test", pyth_feed_id=FEED, historical_latency_seconds=1), client)
            points = await provider.historical_prices(START, START + timedelta(seconds=4), step_seconds=1)
            assert len(points) == 1
            assert points[0].timestamp == START + timedelta(seconds=2)
            assert points[0].available_at == START + timedelta(seconds=3)
            assert points[0].metadata["availability_basis"] == "assumed_historical_latency"
            assert requested[0].endswith(str(int(START.timestamp())))
    asyncio.run(run())


def test_pyth_requires_credentials_and_feed_must_match():
    async def run():
        async with client_for(pyth_payload(feed="b" * 64)) as client:
            with pytest.raises(ProviderError, match="PYTH_API_KEY"):
                await PythProvider(settings(pyth_feed_id=FEED), client).latest_price()
            with pytest.raises(ProviderError, match="requested feed"):
                await PythProvider(settings(pyth_api_key="test", pyth_feed_id=FEED), client).latest_price()
    asyncio.run(run())


def test_pyth_discovery_only_exact_xau_symbol():
    async def run():
        requests = []

        def response(request):
            requests.append(request)
            if request.url.path.endswith("price_feeds"):
                return httpx.Response(200, json=[{"id": FEED, "attributes": {"symbol": "Metal.XAU/USD"}}, {"id": "b" * 64, "attributes": {"symbol": "Metal.XAG/USD"}}])
            return httpx.Response(200, json=pyth_payload())

        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            point = await PythProvider(settings(pyth_api_key="test"), client).latest_price()
            assert point.feed_id == FEED
            assert requests[-1].headers["Authorization"] == "Bearer test"
    asyncio.run(run())


def test_kalshi_settlement_is_gold_price_not_contract_payout():
    async def run():
        async with client_for({"market": market_payload()}) as client:
            point = await KalshiProvider(settings(), client).market("KXGOLD15M-TEST-00")
            assert point.target_price == 4300.1
            assert point.settled_price == 4301.25
            assert point.result_yes is True
            assert point.available_at > point.end_time
    asyncio.run(run())


def test_kalshi_missing_target_does_not_parse_subtitle():
    async def run():
        async with client_for({"market": market_payload(floor_strike=None, yes_sub_title="Target Price: $4300.10", expiration_value="not available")}) as client:
            market = await KalshiProvider(settings(), client).market("KXGOLD15M-TEST-00")
            assert market.target_price is None
            assert market.settled_price is None
    asyncio.run(run())


def test_kalshi_history_preserves_open_and_settled_availability_versions():
    async def run():
        paths = []

        def response(request):
            paths.append(request.url.path)
            return httpx.Response(200, json={"markets": [market_payload()], "cursor": ""})

        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            provider = KalshiProvider(settings(historical_latency_seconds=2), client)
            rows = await provider.historical_markets(START, START + timedelta(hours=1))
            assert len(rows) == 2
            opening, settled = rows
            assert opening.status == "open"
            assert opening.result_yes is None
            assert opening.settled_price is None
            assert opening.settlement_time is None
            assert "result" not in opening.raw and "expiration_value" not in opening.raw
            assert opening.available_at == START + timedelta(seconds=2)
            assert settled.result_yes is True
            assert settled.available_at == START + timedelta(minutes=16, seconds=2)
            assert any("/historical/markets" in path for path in paths)
    asyncio.run(run())


def test_kalshi_history_no_assumption_or_settlement_time_never_backdates_label():
    async def run():
        async with client_for({"markets": [market_payload(settlement_ts=None)], "cursor": ""}) as client:
            provider = KalshiProvider(settings(historical_latency_seconds=0), client)
            rows = await provider.historical_markets(START, START + timedelta(hours=1))
            assert all(row.available_at > row.end_time for row in rows if row.result_yes is not None)
        async with client_for({"markets": [market_payload()], "cursor": ""}) as client:
            rows = await KalshiProvider(settings(), client).historical_markets(START, START + timedelta(hours=1))
            assert all(row.available_at > row.end_time for row in rows)
    asyncio.run(run())


def test_kalshi_book_fractional_depth_and_ed25519_signature(tmp_path):
    key = ed25519.Ed25519PrivateKey.generate()
    path = tmp_path / "key.pem"
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))

    async def run():
        def response(request):
            timestamp = request.headers["KALSHI-ACCESS-TIMESTAMP"]
            signature = base64.b64decode(request.headers["KALSHI-ACCESS-SIGNATURE"])
            key.public_key().verify(signature, f"{timestamp}GET{request.url.path}".encode())
            return httpx.Response(200, json={"orderbook_fp": {"yes_dollars": [["0.4800", "3.50"], ["0.5100", "2.25"]], "no_dollars": [["0.4600", "7.50"]]}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            provider = KalshiProvider(settings(kalshi_api_key="test", kalshi_private_key_path=path), client)
            book = await provider.orderbook("KXGOLD15M-TEST-00")
            assert book.yes_bid == 0.51
            assert book.yes_ask == 0.54
            assert book.no_ask == 0.49
            assert book.yes_ask_size == 7.5
            assert book.no_ask_size == 2.25
    asyncio.run(run())


@pytest.mark.parametrize("payload", [{"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}, {"orderbook": {"yes": [], "no": []}}])
def test_empty_book_is_not_a_free_contract(payload, monkeypatch):
    async def run():
        async with client_for(payload) as client:
            provider = KalshiProvider(settings(), client)
            monkeypatch.setattr(provider, "_auth_headers", lambda *args: {})
            book = await provider.orderbook("KXGOLD15M-TEST-00")
            assert book.yes_ask is None and book.no_ask is None
    asyncio.run(run())


def test_crossed_book_rejected(monkeypatch):
    async def run():
        async with client_for({"orderbook_fp": {"yes_dollars": [["0.7", "3"]], "no_dollars": [["0.6", "4"]]}}) as client:
            provider = KalshiProvider(settings(), client)
            monkeypatch.setattr(provider, "_auth_headers", lambda *args: {})
            with pytest.raises(ProviderError, match="crossed"):
                await provider.orderbook("KXGOLD15M-TEST-00")
    asyncio.run(run())


def test_retry_429_and_no_retry_401(monkeypatch):
    waits = []

    async def sleep(delay):
        waits.append(delay)

    monkeypatch.setattr("gold_model.providers.base.asyncio.sleep", sleep)

    async def run():
        calls = 0

        def response(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={"Retry-After": "1"})
            return httpx.Response(200, json=pyth_payload())

        config = settings(pyth_api_key="test", pyth_feed_id=FEED).model_copy(update={"request_retries": 2})
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            await PythProvider(config, client).latest_price()
            assert calls == 2 and waits == [1]
        async with client_for({}, status=401) as client:
            with pytest.raises(ProviderError, match="credentials"):
                await PythProvider(config, client).latest_price()
        assert waits == [1]
    asyncio.run(run())


def test_databento_history_exact_contract_delay_and_receive_time():
    async def run():
        row = {"hd": {"ts_event": START.isoformat()}, "ts_recv": (START + timedelta(microseconds=100)).isoformat(), "price": "4301.500000000", "symbol": "GCZ5"}

        def response(request):
            assert request.method == "POST"
            assert b"stype_in=raw_symbol" in request.content
            return httpx.Response(200, text=json.dumps(row) + "\n")

        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            provider = DatabentoComexProvider(settings(comex_api_key="test", comex_contract="GCZ5", historical_latency_seconds=0.1), client)
            points = await provider.historical_prices(START, START + timedelta(seconds=5))
            assert points[0].price == 4301.5
            assert points[0].available_at == START + timedelta(microseconds=100100)
            assert points[0].delayed
    asyncio.run(run())


def test_databento_rejects_continuous_and_mismatched_contract():
    async def run():
        async with client_for({}) as client:
            provider = DatabentoComexProvider(settings(comex_api_key="test", comex_contract="GC.c.0"), client)
            with pytest.raises(ProviderError, match="continuous"):
                await provider.historical_prices(START, START + timedelta(seconds=5))
            provider = DatabentoComexProvider(settings(comex_api_key="test", comex_contract="GCZ5"), client)
            with pytest.raises(ProviderError, match="unexpected contract"):
                provider._history_point({"symbol": "GCG6"}, datetime.now(UTC))
    asyncio.run(run())


def test_csv_requires_explicit_available_at_delay_and_contract(tmp_path):
    path = tmp_path / "comex.csv"
    path.write_text("timestamp,available_at,price,contract,delayed\n2025-01-02T14:00:00Z,2025-01-02T14:00:01Z,4300.5,GCZ5,true\n")

    async def run():
        provider = CSVComexProvider(settings(comex_provider="csv", comex_contract="GCZ5", comex_csv_path=path))
        try:
            point = await provider.latest_price()
            assert point.available_at == START + timedelta(seconds=1)
            assert point.delayed
            path.write_text("timestamp,available_at,price,contract,delayed\n2025-01-02T14:00:00,2025-01-02T14:00:01Z,4300.5,GCZ5,true\n")
            with pytest.raises(ProviderError, match="row 2"):
                await provider.latest_price()
            path.write_text("timestamp,price,contract,delayed\n2025-01-02T14:00:00Z,4300.5,GCZ5,true\n")
            with pytest.raises(ProviderError, match="columns"):
                await provider.latest_price()
        finally:
            await provider.aclose()
    asyncio.run(run())


def test_live_databento_keeps_original_receipt_on_repeated_poll(monkeypatch):
    class Trade:
        ts_event = int(START.timestamp()) * 1_000_000_000
        ts_recv = ts_event + 100000
        price = 4301500000000
        size = 2
        instrument_id = 1
        sequence = 7

    class Live:
        def __init__(self, **kwargs):
            self.callback = None

        def subscribe(self, **kwargs):
            assert kwargs["symbols"] == ["GCZ5"]
            assert kwargs["stype_in"] == "raw_symbol"

        def add_callback(self, callback, **kwargs):
            self.callback = callback

        def start(self):
            self.callback(Trade())

        def stop(self):
            pass

        def block_for_close(self, timeout):
            pass

    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace(Live=Live, TradeMsg=Trade, ErrorMsg=type("Error", (), {})))
    raw = []

    async def run():
        provider = DatabentoComexProvider(settings(comex_api_key="test", comex_contract="GCZ5"), raw_sink=lambda *args: raw.append(args))
        try:
            first = await provider.latest_price()
            second = await provider.latest_price()
            assert first.available_at == second.available_at
            assert first.price == pytest.approx(4301.5)
            assert first.delayed is False
            assert len(raw) == 1
        finally:
            await provider.aclose()
    asyncio.run(run())


def test_nanosecond_availability_rounds_up_never_down():
    assert utc("2025-01-02T14:00:00.000000001Z") == START + timedelta(microseconds=1)
    with pytest.raises(ProviderError, match="UTC offset"):
        utc("2025-01-02T14:00:00")
