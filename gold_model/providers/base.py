"""Small, read-only provider contracts and shared HTTP failure handling."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

import httpx

from gold_model.config import Settings
from gold_model.data.models import PricePoint

RawSink = Callable[[str, str, dict[str, Any], datetime], None]
logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """An actionable provider failure, without credentials or response bodies."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class PriceProvider(Protocol):
    async def latest_price(self) -> PricePoint: ...

    async def historical_prices(self, start: datetime, end: datetime) -> list[PricePoint]: ...

    async def aclose(self) -> None: ...


def utc(value: datetime | str) -> datetime:
    result = (
        datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    )
    if result.tzinfo is None or result.utcoffset() is None:
        raise ProviderError("Provider timestamp must have an explicit UTC offset")
    if isinstance(value, str):
        fraction = re.search(r"\.([0-9]{7,})", value)
        if fraction and any(digit != "0" for digit in fraction.group(1)[6:]):
            result += timedelta(microseconds=1)
    return result.astimezone(UTC)


def secret(value: Any) -> str:
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value or "")


def historical_availability(
    timestamp: datetime, received_at: datetime, latency: float | None
) -> tuple[datetime, dict[str, Any]]:
    """Never silently pretend retrospectively downloaded data was recorded live."""
    if latency is None:
        return max(timestamp, received_at), {"availability_basis": "download_time"}
    return timestamp + timedelta(seconds=latency), {
        "availability_basis": "assumed_historical_latency",
        "historical_latency_seconds": latency,
        "downloaded_at": received_at.isoformat(),
    }


class HTTPProvider:
    provider_name = "http"

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        raw_sink: RawSink | None = None,
    ) -> None:
        self.settings = settings
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=settings.request_timeout_seconds)
        self.raw_sink = raw_sink
        self._rate_lock = asyncio.Lock()
        self._next_request_at = 0.0

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def record_raw(self, kind: str, payload: dict[str, Any], received_at: datetime) -> None:
        if self.raw_sink is not None:
            self.raw_sink(self.provider_name, kind, payload, received_at)

    async def _pace(self) -> None:
        async with self._rate_lock:
            delay = self._next_request_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_request_at = time.monotonic() + self.settings.provider_min_interval_seconds

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers_factory: Callable[[], dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> tuple[httpx.Response, datetime]:
        """Retry transport, 429 and 5xx failures; never retry authorization errors."""
        attempts = self.settings.request_retries + 1
        for attempt in range(attempts):
            await self._pace()
            if headers_factory is not None:
                kwargs["headers"] = headers_factory()
            response = None
            try:
                response = await self.client.request(method, url, **kwargs)
            except httpx.TransportError:
                if attempt + 1 == attempts:
                    raise ProviderError(
                        f"{self.provider_name}: transport failed after {attempts} attempts"
                    ) from None
            else:
                if response.is_success:
                    return response, datetime.now(UTC)
                retryable = response.status_code == 429 or response.status_code >= 500
                if not retryable or attempt + 1 == attempts:
                    hint = (
                        " Check credentials and data entitlements."
                        if response.status_code in (401, 403)
                        else ""
                    )
                    raise ProviderError(
                        f"{self.provider_name}: HTTP {response.status_code}.{hint}",
                        status_code=response.status_code,
                    )
            delay = min(30.0, 0.5 * 2**attempt)
            if response is not None and response.headers.get("Retry-After"):
                retry_after = response.headers["Retry-After"]
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    try:
                        delay = max(
                            delay,
                            (
                                parsedate_to_datetime(retry_after) - datetime.now(UTC)
                            ).total_seconds(),
                        )
                    except (TypeError, ValueError):
                        pass
                # A long server-directed pause belongs to the caller's next cycle.
                if delay > 60:
                    raise ProviderError(
                        f"{self.provider_name}: rate limited; retry after {delay:.0f}s", 429
                    )
            logger.warning(
                "provider_retry",
                extra={"context": {"provider": self.provider_name, "attempt": attempt + 1}},
            )
            await asyncio.sleep(delay)
        raise AssertionError("Unreachable retry state")

    async def get_json(self, url: str, *, kind: str, **kwargs: Any) -> tuple[Any, datetime]:
        response, received_at = await self._request("GET", url, **kwargs)
        try:
            payload = response.json()
        except ValueError:
            self.record_raw(kind, {"text": response.text}, received_at)
            raise ProviderError(f"{self.provider_name}: malformed JSON response") from None
        self.record_raw(
            kind, payload if isinstance(payload, dict) else {"data": payload}, received_at
        )
        return payload, received_at
