"""Read-only Kalshi market discovery, settlement metadata, and executable books."""

from __future__ import annotations

import base64
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from gold_model.data.models import BookSnapshot, MarketWindow
from gold_model.providers.base import (
    HTTPProvider,
    ProviderError,
    historical_availability,
    secret,
    utc,
)

logger = logging.getLogger(__name__)


class KalshiProvider(HTTPProvider):
    provider_name = "kalshi"

    def _url(self, path: str) -> str:
        return f"{self.settings.kalshi_base_url.rstrip('/')}{path}"

    def _auth_headers(self, path: str, required: bool = False) -> dict[str, str]:
        key_id = secret(self.settings.kalshi_api_key)
        key_path = self.settings.kalshi_private_key_path
        if not key_id and not key_path and not required:
            return {}
        if not key_id or not key_path:
            raise ProviderError(
                "Kalshi order books require KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH"
            )
        if not hasattr(self, "_private_key"):
            try:
                self._private_key = serialization.load_pem_private_key(
                    Path(key_path).read_bytes(), password=None
                )
            except (OSError, ValueError, TypeError):
                raise ProviderError(
                    "Unable to read Kalshi private key; use an unencrypted RSA or Ed25519 PEM file"
                ) from None
        timestamp = str(int(datetime.now(UTC).timestamp() * 1000))
        signing_path = urlparse(self._url(path)).path
        message = f"{timestamp}GET{signing_path}".encode()
        if isinstance(self._private_key, ed25519.Ed25519PrivateKey):
            signature = self._private_key.sign(message)
        elif isinstance(self._private_key, rsa.RSAPrivateKey):
            signature = self._private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256(),
            )
        else:
            raise ProviderError("Kalshi key must be RSA or Ed25519")
        return {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }

    async def _get(
        self, path: str, *, kind: str, required_auth: bool = False, **kwargs: Any
    ) -> tuple[Any, datetime]:
        return await self.get_json(
            self._url(path),
            kind=kind,
            headers_factory=lambda: self._auth_headers(path, required_auth),
            **kwargs,
        )

    def _normalize_market(
        self, row: dict[str, Any], received_at: datetime, *, historical: bool
    ) -> MarketWindow:
        try:
            start = utc(row["open_time"])
            end = utc(row["close_time"])
            market_id = str(row["ticker"])
        except (KeyError, ValueError, TypeError):
            raise ProviderError(
                "Kalshi market is missing ticker or aware open/close timestamps"
            ) from None
        if not market_id.startswith(f"{self.settings.kalshi_series_ticker}-"):
            raise ProviderError("Kalshi returned a market outside the configured gold series")
        if (end - start).total_seconds() != 900:
            raise ProviderError(
                f"Kalshi market {market_id} is not a 15-minute open/close window; inspect its rules"
            )
        strike_type = row.get("strike_type")
        target = None
        if strike_type == "greater_or_equal" and row.get("floor_strike") is not None:
            target = float(row["floor_strike"])
        elif row.get("floor_strike") is not None:
            raise ProviderError(
                f"Unsupported Kalshi strike_type={strike_type!r}; cannot treat as at-or-above"
            )
        result = {"yes": True, "no": False}.get(row.get("result", ""))
        # expiration_value is the underlying reference value. settlement_value
        # and settlement_value_dollars are contract payouts, never gold prices.
        settled_price = None
        if result is not None and row.get("expiration_value") not in (None, ""):
            try:
                parsed = float(row["expiration_value"])
                if math.isfinite(parsed) and parsed > 0:
                    settled_price = parsed
            except (TypeError, ValueError):
                pass
        raw = dict(row)
        settlement_time = utc(row["settlement_ts"]) if row.get("settlement_ts") else None
        available_at = received_at
        if historical:
            raw["_availability"] = {"availability_basis": "download_time"}
            if result is not None and settlement_time is not None:
                available_at, basis = historical_availability(
                    max(end, settlement_time), received_at, self.settings.historical_latency_seconds
                )
                raw["_availability"] = basis
        return MarketWindow(
            market_id=market_id,
            event_id=row.get("event_ticker"),
            start_time=start,
            end_time=end,
            available_at=available_at,
            target_price=target,
            status=str(row.get("status", "unknown")),
            result_yes=result,
            settled_price=settled_price,
            settlement_time=settlement_time,
            raw=raw,
        )

    def _historical_versions(
        self, market: MarketWindow, received_at: datetime
    ) -> list[MarketWindow]:
        """Separate an assumed opening target from the subsequently known label."""
        latency = self.settings.historical_latency_seconds
        if latency is None or market.target_price is None:
            return [market]
        available_at, basis = historical_availability(market.start_time, received_at, latency)
        # Do not include the retrospective final response in the assumed opening
        # version: even raw fields must not suggest an outcome known at open.
        raw = {
            "ticker": market.market_id,
            "event_ticker": market.event_id,
            "open_time": market.start_time.isoformat(),
            "close_time": market.end_time.isoformat(),
            "floor_strike": market.target_price,
            "status": "open",
            "_availability": {**basis, "assumption": "final_target_assumed_known_at_open"},
        }
        opening = market.model_copy(
            update={
                "available_at": available_at,
                "status": "open",
                "result_yes": None,
                "settled_price": None,
                "settlement_time": None,
                "raw": raw,
            }
        )
        return [opening, market]

    async def _pages(
        self, path: str, params: dict[str, Any], *, historical: bool
    ) -> list[MarketWindow]:
        markets = []
        seen_cursors: set[str] = set()
        while True:
            payload, received_at = await self._get(
                path, kind="historical_markets" if historical else "markets", params=params
            )
            for row in payload.get("markets", []):
                try:
                    market = self._normalize_market(row, received_at, historical=historical)
                    markets.extend(
                        self._historical_versions(market, received_at) if historical else [market]
                    )
                except ProviderError as exc:
                    logger.warning(
                        "market_rejected",
                        extra={
                            "context": {
                                "provider": "kalshi",
                                "reason": str(exc),
                                "market_id": row.get("ticker"),
                            }
                        },
                    )
            cursor = payload.get("cursor")
            if not cursor:
                return markets
            if cursor in seen_cursors:
                raise ProviderError(
                    "Kalshi repeated a pagination cursor; stopped to avoid an infinite collection loop"
                )
            seen_cursors.add(cursor)
            params = {**params, "cursor": cursor}

    async def active_markets(self) -> list[MarketWindow]:
        markets = await self._pages(
            "/markets",
            {"series_ticker": self.settings.kalshi_series_ticker, "status": "open", "limit": 1000},
            historical=False,
        )
        now = datetime.now(UTC)
        return sorted(
            (market for market in markets if market.start_time <= now < market.end_time),
            key=lambda market: market.end_time,
        )

    async def historical_markets(self, start: datetime, end: datetime) -> list[MarketWindow]:
        """Combine recent and archived markets; archived API has no date filters."""
        start, end = utc(start), utc(end)
        if end <= start:
            raise ValueError("Market history requires start < end")
        common = {"series_ticker": self.settings.kalshi_series_ticker, "limit": 1000}
        recent = await self._pages(
            "/markets",
            {
                **common,
                "min_close_ts": int(start.timestamp()),
                "max_close_ts": int(end.timestamp()),
            },
            historical=True,
        )
        archived = await self._pages("/historical/markets", common, historical=True)
        unique = {
            (market.market_id, market.available_at, market.status): market
            for market in archived + recent
            if start <= market.end_time < end
        }
        return sorted(unique.values(), key=lambda market: (market.end_time, market.available_at))

    async def market(self, market_id: str) -> MarketWindow:
        path = f"/markets/{quote(market_id, safe='')}"
        try:
            payload, received_at = await self._get(path, kind="market")
        except ProviderError as exc:
            if exc.status_code != 404:
                raise
            payload, received_at = await self._get(f"/historical{path}", kind="historical_market")
        # Even resolved metadata is only known when fetched unless history was
        # explicitly requested with a documented availability assumption.
        return self._normalize_market(payload["market"], received_at, historical=False)

    @staticmethod
    def _levels(raw: Any, *, cents: bool) -> list[list[float]]:
        levels: dict[float, float] = {}
        for row in raw or []:
            if len(row) != 2:
                raise ProviderError("Kalshi returned malformed order-book depth")
            price, size = float(row[0]) / (100 if cents else 1), float(row[1])
            if (
                not math.isfinite(price)
                or not 0 <= price <= 1
                or not math.isfinite(size)
                or size < 0
            ):
                raise ProviderError("Kalshi returned invalid probability or size in order book")
            if size > 0:
                levels[price] = levels.get(price, 0) + size
        return [[price, levels[price]] for price in sorted(levels, reverse=True)]

    async def orderbook(self, market_id: str) -> BookSnapshot:
        path = f"/markets/{quote(market_id, safe='')}/orderbook"
        try:
            payload, received_at = await self._get(path, kind="orderbook", required_auth=True)
        except ProviderError as exc:
            if exc.status_code != 404:
                raise
            received_at = datetime.now(UTC)
            payload = {"missing_book": True, "status_code": 404}
            self.record_raw("orderbook", payload, received_at)
        if "orderbook_fp" in payload:
            book = payload["orderbook_fp"] or {}
            yes = self._levels(book.get("yes_dollars"), cents=False)
            no = self._levels(book.get("no_dollars"), cents=False)
        elif "orderbook" in payload:
            book = payload["orderbook"] or {}
            yes = self._levels(book.get("yes"), cents=True)
            no = self._levels(book.get("no"), cents=True)
        elif payload.get("missing_book"):
            yes, no = [], []
        else:
            raise ProviderError("Kalshi response has no recognized order-book schema")
        if yes and no and yes[0][0] + no[0][0] > 1 + 1e-10:
            raise ProviderError("Kalshi returned a crossed order book; no executable quote")
        return BookSnapshot(
            timestamp=received_at,
            available_at=received_at,
            market_id=market_id,
            yes_bid=yes[0][0] if yes else None,
            no_bid=no[0][0] if no else None,
            yes_ask=round(1 - no[0][0], 10) if no else None,
            no_ask=round(1 - yes[0][0], 10) if yes else None,
            yes_ask_size=no[0][1] if no else None,
            no_ask_size=yes[0][1] if yes else None,
            yes_bids=yes,
            no_bids=no,
            raw=payload,
        )

    async def recent_trades(self, market_id: str, limit: int = 100) -> list[dict[str, Any]]:
        payload, _ = await self._get(
            "/markets/trades",
            kind="trades",
            params={"ticker": market_id, "limit": min(max(limit, 1), 1000)},
        )
        return payload.get("trades", [])
