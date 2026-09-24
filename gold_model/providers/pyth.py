"""Pyth Hermes XAU/USD polling and timestamp history, with explicit availability."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from gold_model.data.models import PricePoint
from gold_model.providers.base import (
    HTTPProvider,
    ProviderError,
    historical_availability,
    secret,
    utc,
)


class PythProvider(HTTPProvider):
    provider_name = "pyth"

    def _headers(self) -> dict[str, str]:
        key = secret(self.settings.pyth_api_key)
        if not key:
            raise ProviderError("Pyth requires PYTH_API_KEY; obtain a Pyth data plan/API key")
        return {"Authorization": f"Bearer {key}"}

    async def _feed_id(self) -> str:
        configured = self.settings.pyth_feed_id
        if configured:
            if not re.fullmatch(r"(?:0x)?[0-9a-fA-F]{64}", configured):
                raise ProviderError("PYTH_FEED_ID must be a 32-byte hexadecimal feed identifier")
            return configured.removeprefix("0x").lower()
        if hasattr(self, "_discovered_feed_id"):
            return self._discovered_feed_id
        payload, _ = await self.get_json(
            f"{self.settings.pyth_base_url.rstrip('/')}/v2/price_feeds",
            kind="feed_metadata",
            headers=self._headers(),
            params={"query": "XAU/USD", "asset_type": "metal"},
        )
        matches = [
            row["id"]
            for row in payload
            if row.get("attributes", {}).get("symbol", "").split(".")[-1] == "XAU/USD"
        ]
        if len(matches) != 1:
            raise ProviderError(
                "Pyth XAU/USD feed discovery was ambiguous or empty; configure PYTH_FEED_ID"
            )
        self._discovered_feed_id = str(matches[0]).removeprefix("0x").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", self._discovered_feed_id):
            raise ProviderError("Pyth feed discovery returned an invalid feed identifier")
        return self._discovered_feed_id

    def _normalize(
        self, payload: dict[str, Any], received_at: datetime, feed_id: str, *, historical: bool
    ) -> PricePoint:
        rows = [
            row
            for row in payload.get("parsed", [])
            if str(row.get("id", "")).removeprefix("0x").lower() == feed_id
        ]
        if len(rows) != 1:
            raise ProviderError("Pyth response does not contain exactly one requested feed")
        row = rows[0]
        try:
            value = row["price"]
            scale = Decimal(10) ** int(value["expo"])
            price = float(Decimal(value["price"]) * scale)
            confidence = float(Decimal(value["conf"]) * scale)
            timestamp = datetime.fromtimestamp(int(value["publish_time"]), UTC)
            if (
                not math.isfinite(price)
                or price <= 0
                or not math.isfinite(confidence)
                or confidence < 0
            ):
                raise ValueError("invalid price")
        except (KeyError, TypeError, ValueError, ArithmeticError, OverflowError):
            raise ProviderError("Pyth returned an invalid price observation") from None
        if timestamp > received_at + timedelta(seconds=5):
            raise ProviderError("Pyth publish time is in the future; check local clock")
        if historical:
            available_at, metadata = historical_availability(
                timestamp, received_at, self.settings.historical_latency_seconds
            )
        else:
            available_at, metadata = (
                max(timestamp, received_at),
                {"availability_basis": "local_receive_time"},
            )
        metadata["source_metadata"] = row.get("metadata", {})
        return PricePoint(
            timestamp=timestamp,
            available_at=available_at,
            price=price,
            confidence=confidence,
            provider=self.provider_name,
            feed_id=feed_id,
            metadata=metadata,
            raw=row,
        )

    async def latest_price(self) -> PricePoint:
        feed_id = await self._feed_id()
        payload, received_at = await self.get_json(
            f"{self.settings.pyth_base_url.rstrip('/')}/v2/updates/price/latest",
            kind="latest_price",
            headers=self._headers(),
            params={"ids[]": feed_id, "parsed": "true"},
        )
        return self._normalize(payload, received_at, feed_id, historical=False)

    async def historical_prices(
        self, start: datetime, end: datetime, step_seconds: int = 5
    ) -> list[PricePoint]:
        """Sample the documented timestamp API; retain actual returned publish times.

        Hermes may return the first update at/after a requested timestamp. The
        observation is never backdated to the query time. Retention depends on the
        configured deployment and plan; unsupported history fails explicitly.
        """
        start, end = utc(start), utc(end)
        if end <= start or step_seconds < 1:
            raise ValueError("History requires start < end and step_seconds >= 1")
        feed_id = await self._feed_id()
        prices: dict[datetime, PricePoint] = {}
        query_time = start
        while query_time < end:
            payload, received_at = await self.get_json(
                f"{self.settings.pyth_base_url.rstrip('/')}/v2/updates/price/{int(query_time.timestamp())}",
                kind="historical_price",
                headers=self._headers(),
                params={"ids[]": feed_id, "parsed": "true"},
            )
            point = self._normalize(payload, received_at, feed_id, historical=True)
            if start <= point.timestamp < end:
                prices[point.timestamp] = point
            query_time += timedelta(seconds=step_seconds)
        return sorted(prices.values(), key=lambda point: point.timestamp)
