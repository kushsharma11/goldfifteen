"""Exact-contract COMEX data through Databento or explicit-availability CSV."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from gold_model.data.models import PricePoint
from gold_model.providers.base import (
    HTTPProvider,
    ProviderError,
    historical_availability,
    secret,
    utc,
)

logger = logging.getLogger(__name__)


def _contract(settings: Any) -> str:
    contract = settings.comex_contract
    if not contract or not re.fullmatch(r"GC[FGHJKMNQUVXZ]\d{1,2}", contract):
        raise ProviderError(
            "COMEX_CONTRACT must name one GC futures expiry, e.g. GCZ6; continuous/root symbols are prohibited"
        )
    return contract


def _nanoseconds(value: int) -> datetime:
    seconds, nanos = divmod(int(value), 1_000_000_000)
    # Round UP to a microsecond: normalization must never make data look earlier.
    return datetime.fromtimestamp(seconds, UTC) + timedelta(microseconds=(nanos + 999) // 1000)


class DatabentoComexProvider(HTTPProvider):
    """Databento trades, with independent historical HTTP and optional live SDK.

    COMEX_PROVIDER=databento requires the live extra and exchange entitlements.
    databento_delayed polls historical trades and always labels them delayed.
    """

    provider_name = "databento"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._live: Any = None
        self._consumer: asyncio.Task[None] | None = None
        self._live_error: ProviderError | None = None
        self._latest: PricePoint | None = None
        self._ready = asyncio.Event()
        self._queue: asyncio.Queue[tuple[dict[str, Any], datetime]] = asyncio.Queue(maxsize=10000)
        self._start_lock = asyncio.Lock()

    def _key(self) -> str:
        key = secret(self.settings.comex_api_key)
        if not key:
            raise ProviderError(
                "Databento requires COMEX_API_KEY and COMEX/CME market-data entitlements"
            )
        return key

    def _history_point(self, row: dict[str, Any], received_at: datetime) -> PricePoint:
        contract = _contract(self.settings)
        if row.get("symbol") != contract:
            raise ProviderError(
                "Databento returned an unexpected contract; refusing mixed-contract history"
            )
        try:
            timestamp = utc(row["hd"]["ts_event"])
            source_receive = utc(row["ts_recv"])
            price = float(row["price"])
        except (KeyError, TypeError, ValueError):
            raise ProviderError("Databento returned invalid trade data") from None
        available_at, metadata = historical_availability(
            max(timestamp, source_receive), received_at, self.settings.historical_latency_seconds
        )
        metadata.update(
            {
                "source_receive_time": source_receive.isoformat(),
                "dataset": self.settings.comex_dataset,
                "schema": "trades",
            }
        )
        return PricePoint(
            timestamp=timestamp,
            available_at=available_at,
            price=price,
            contract=contract,
            provider=self.provider_name,
            delayed=True,
            metadata=metadata,
            raw=row,
        )

    async def historical_prices(self, start: datetime, end: datetime) -> list[PricePoint]:
        start, end = utc(start), utc(end)
        if end <= start:
            raise ValueError("COMEX history requires start < end")
        points = []
        chunk_start = start
        # Bound each HTTP response to one hour. The historical endpoint is a
        # documented read-only RPC using POST; no trading endpoint exists here.
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(hours=1), end)
            response, received_at = await self._request(
                "POST",
                f"{self.settings.databento_base_url.rstrip('/')}/timeseries.get_range",
                auth=(self._key(), ""),
                data={
                    "dataset": self.settings.comex_dataset,
                    "schema": "trades",
                    "symbols": _contract(self.settings),
                    "stype_in": "raw_symbol",
                    "start": chunk_start.isoformat(),
                    "end": chunk_end.isoformat(),
                    "encoding": "json",
                    "pretty_px": "true",
                    "pretty_ts": "true",
                    "map_symbols": "true",
                },
            )
            self.record_raw(
                "historical_trades",
                {
                    "text": response.text,
                    "start": chunk_start.isoformat(),
                    "end": chunk_end.isoformat(),
                },
                received_at,
            )
            try:
                rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
            except ValueError:
                raise ProviderError(
                    "Databento returned malformed JSON-lines trade history"
                ) from None
            for row in rows:
                point = self._history_point(row, received_at)
                if start <= point.timestamp < end:
                    points.append(point)
            chunk_start = chunk_end
        return sorted(points, key=lambda point: point.timestamp)

    async def _consume_live(self) -> None:
        try:
            while True:
                raw, received_at = await self._queue.get()
                self.record_raw("live_trade", raw, received_at)
                timestamp = _nanoseconds(raw["ts_event"])
                if timestamp > received_at + timedelta(seconds=5):
                    raise ProviderError(
                        "Databento trade timestamp is in the future; check local clock"
                    )
                point = PricePoint(
                    timestamp=timestamp,
                    available_at=max(timestamp, received_at),
                    price=raw["price"] * 1e-9,
                    contract=_contract(self.settings),
                    provider=self.provider_name,
                    delayed=False,
                    metadata={
                        "availability_basis": "local_receive_time",
                        "dataset": self.settings.comex_dataset,
                        "schema": "trades",
                        "source_receive_time": _nanoseconds(raw["ts_recv"]).isoformat(),
                    },
                    raw=raw,
                )
                if self._latest is None or point.timestamp >= self._latest.timestamp:
                    self._latest = point
                self._ready.set()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._set_live_error(
                "Databento live normalization or raw persistence failed; inspect data/storage"
            )

    def _set_live_error(self, reason: str) -> None:
        self._live_error = ProviderError(reason)
        self._ready.set()

    def _enqueue(self, raw: dict[str, Any], received_at: datetime) -> None:
        try:
            self._queue.put_nowait((raw, received_at))
        except asyncio.QueueFull:
            self._set_live_error(
                "Databento live capture queue overflowed; data was lost, so predictions are disabled"
            )

    async def _ensure_live(self) -> None:
        async with self._start_lock:
            if self._live is not None:
                return
            contract, key = _contract(self.settings), self._key()
            try:
                import databento as db
            except ImportError:
                raise ProviderError(
                    "Databento live support requires pip install -e '.[comex]' (or select COMEX_PROVIDER=csv)"
                ) from None
            loop = asyncio.get_running_loop()

            def on_record(record: Any) -> None:
                received_at = datetime.now(UTC)
                if isinstance(record, db.TradeMsg):
                    raw = {
                        name: int(getattr(record, name))
                        for name in (
                            "ts_event",
                            "ts_recv",
                            "price",
                            "size",
                            "instrument_id",
                            "sequence",
                        )
                    }
                    for name in ("publisher_id", "flags", "depth", "ts_in_delta"):
                        if hasattr(record, name):
                            raw[name] = int(getattr(record, name))
                    for name in ("action", "side"):
                        if hasattr(record, name):
                            raw[name] = str(getattr(record, name))
                    raw["symbol"] = contract
                    loop.call_soon_threadsafe(self._enqueue, raw, received_at)
                elif isinstance(record, db.ErrorMsg):
                    loop.call_soon_threadsafe(
                        self._set_live_error,
                        "Databento live gateway reported an error; verify subscription, contract and entitlements",
                    )

            def on_error(_error: Exception) -> None:
                loop.call_soon_threadsafe(
                    self._set_live_error, "Databento live callback failed; collection stopped"
                )

            def start_client() -> Any:
                client = db.Live(key=key, reconnect_policy="reconnect")
                try:
                    client.subscribe(
                        dataset=self.settings.comex_dataset,
                        schema="trades",
                        stype_in="raw_symbol",
                        symbols=[contract],
                    )
                    client.add_callback(on_record, exception_callback=on_error)
                    client.start()
                except Exception:
                    try:
                        client.terminate()
                    except Exception:
                        pass
                    raise
                return client

            self._consumer = asyncio.create_task(self._consume_live())
            try:
                self._live = await asyncio.to_thread(start_client)
            except Exception:
                self._consumer.cancel()
                await asyncio.gather(self._consumer, return_exceptions=True)
                self._consumer = None
                raise ProviderError(
                    "Databento live subscription failed; check COMEX_API_KEY, contract and exchange entitlements"
                ) from None

    async def latest_price(self) -> PricePoint:
        if self.settings.comex_provider == "databento_delayed":
            end = datetime.now(UTC) - timedelta(seconds=self.settings.comex_delay_seconds)
            rows = await self.historical_prices(end - timedelta(minutes=5), end)
            if not rows:
                raise ProviderError("No recent Databento historical trades; market may be closed")
            # Historical availability assumptions must never make a delayed
            # polling observation available before this actual download.
            latest = rows[-1]
            return latest.model_copy(update={"available_at": datetime.now(UTC), "delayed": True})
        await self._ensure_live()
        if self._latest is None and self._live_error is None:
            try:
                await asyncio.wait_for(
                    self._ready.wait(), timeout=self.settings.request_timeout_seconds
                )
            except TimeoutError:
                raise ProviderError(
                    "No Databento live trade arrived before timeout; market may be closed"
                ) from None
        if self._live_error:
            raise self._live_error
        if self._latest is None:
            raise ProviderError("Databento live stream has no usable trade")
        return self._latest

    async def aclose(self) -> None:
        if self._live is not None:
            try:
                await asyncio.to_thread(self._live.stop)
                await asyncio.to_thread(self._live.block_for_close, timeout=5)
            except Exception:
                logger.warning(
                    "comex_shutdown_failed", extra={"context": {"provider": self.provider_name}}
                )
            self._live = None
        if self._consumer is not None:
            self._consumer.cancel()
            await asyncio.gather(self._consumer, return_exceptions=True)
            self._consumer = None
        await super().aclose()


class CSVComexProvider(HTTPProvider):
    """Read an externally maintained CSV containing actual receipt timestamps."""

    provider_name = "comex_csv"

    async def _read(self) -> list[PricePoint]:
        path = self.settings.comex_csv_path
        if not path:
            raise ProviderError("COMEX_PROVIDER=csv requires COMEX_CSV_PATH")
        contract = _contract(self.settings)
        try:
            text = await asyncio.to_thread(Path(path).read_text, encoding="utf-8")
        except OSError:
            raise ProviderError("Cannot read COMEX_CSV_PATH") from None
        received_at = datetime.now(UTC)
        self.record_raw("csv", {"text": text, "path": str(path)}, received_at)
        reader = csv.DictReader(text.splitlines())
        required = {"timestamp", "available_at", "price", "contract", "delayed"}
        if not required.issubset(reader.fieldnames or []):
            raise ProviderError(f"COMEX CSV requires columns {', '.join(sorted(required))}")
        points = []
        for line_number, row in enumerate(reader, start=2):
            try:
                if row["contract"] != contract:
                    raise ValueError("mixed contract")
                delayed = row["delayed"].strip().lower()
                if delayed not in ("true", "false", "1", "0"):
                    raise ValueError("delayed must be explicit")
                timestamp, available_at = utc(row["timestamp"]), utc(row["available_at"])
                if available_at < timestamp:
                    raise ValueError("available_at before timestamp")
                points.append(
                    PricePoint(
                        timestamp=timestamp,
                        available_at=available_at,
                        price=float(row["price"]),
                        contract=contract,
                        provider=self.provider_name,
                        delayed=delayed in ("true", "1"),
                        metadata={
                            "availability_basis": "imported_recorded_receipt",
                            "imported_at": received_at.isoformat(),
                        },
                        raw=row,
                    )
                )
            except (TypeError, ValueError, ProviderError):
                raise ProviderError(
                    f"Invalid COMEX CSV row {line_number}; require one contract, aware timestamps, positive price and explicit delayed flag"
                ) from None
        return sorted(points, key=lambda point: point.timestamp)

    async def latest_price(self) -> PricePoint:
        now = datetime.now(UTC)
        points = [
            point
            for point in await self._read()
            if point.timestamp <= now and point.available_at <= now
        ]
        if not points:
            raise ProviderError("COMEX CSV contains no observation available now")
        return points[-1]

    async def historical_prices(self, start: datetime, end: datetime) -> list[PricePoint]:
        start, end = utc(start), utc(end)
        if end <= start:
            raise ValueError("COMEX history requires start < end")
        return [point for point in await self._read() if start <= point.timestamp < end]


def create_comex_provider(
    settings: Any, **kwargs: Any
) -> DatabentoComexProvider | CSVComexProvider:
    if settings.comex_provider in ("databento", "databento_delayed"):
        return DatabentoComexProvider(settings, **kwargs)
    if settings.comex_provider == "csv":
        return CSVComexProvider(settings, **kwargs)
    raise ProviderError(
        "COMEX is disabled; configure COMEX_PROVIDER=databento, databento_delayed or csv"
    )


# Conventional mixed-case spelling for callers; both names denote one class.
CsvComexProvider = CSVComexProvider
