"""Read-only collection; one failed feed never fabricates values for another."""

import asyncio
import logging
from datetime import datetime
from typing import Any

from gold_model.config import Settings
from gold_model.data.database import Database
from gold_model.utils.time import utc_now

logger = logging.getLogger(__name__)


class Collector:
    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database
        self.providers: dict[str, Any] = {}
        self._last_settlement_refresh: datetime | None = None
        self._active_ids: set[str] = set()

    def provider(self, name: str) -> Any:
        if name not in self.providers:
            from gold_model.providers.comex import CsvComexProvider, DatabentoComexProvider
            from gold_model.providers.kalshi import KalshiProvider
            from gold_model.providers.pyth import PythProvider

            cls = {"pyth": PythProvider, "kalshi": KalshiProvider, "comex": DatabentoComexProvider}[name]
            if name == "comex" and self.settings.comex_provider == "disabled":
                raise ValueError("COMEX disabled by configuration")
            if name == "comex" and self.settings.comex_provider == "csv":
                cls = CsvComexProvider
            self.providers[name] = cls(self.settings, raw_sink=self.database.raw_sink)
        return self.providers[name]

    async def collect_once(self, source: str = "all") -> dict:
        names = ["pyth", "kalshi", "comex"] if source == "all" else [source]
        if any(name not in ("pyth", "kalshi", "comex") for name in names):
            raise ValueError("Source must be pyth, kalshi, comex, or all")

        async def collect(name: str) -> tuple[str, dict]:
            try:
                if name == "comex" and self.settings.comex_provider == "disabled":
                    return name, {"disabled": True, "reason": "COMEX_PROVIDER=disabled"}
                provider = self.provider(name)
                if name != "kalshi":
                    point = await provider.latest_price()
                    added = self.database.add_price(point, "spot" if name == "pyth" else "comex")
                    age = (utc_now() - point.timestamp).total_seconds()
                    max_age = self.settings.spot_max_age_seconds if name == "pyth" else self.settings.comex_max_age_seconds
                    if age > max_age or point.delayed:
                        logger.warning("stale_data", extra={"context": {"provider": name, "age_seconds": age, "delayed": point.delayed}})
                    return name, {"observations": int(added), "age_seconds": age, "delayed": point.delayed}
                markets = await provider.active_markets()
                active_ids = {m.market_id for m in markets}
                if active_ids != self._active_ids:
                    logger.info("market_transition", extra={"context": {"active": sorted(active_ids)}})
                    self._active_ids = active_ids
                books, failures = 0, []
                for market in markets:
                    self.database.add_market(market)
                    try:
                        book = await provider.orderbook(market.market_id)
                        books += int(self.database.add_book(book))
                    except Exception as exc:
                        failures.append(f"{market.market_id}: {exc}")
                        logger.warning("orderbook_unavailable", extra={"context": {"market_id": market.market_id, "error": str(exc)}})
                now = utc_now()
                if self._last_settlement_refresh is None or (now-self._last_settlement_refresh).total_seconds() >= 60:
                    for old in sorted(self.database.pending_markets(now), key=lambda m: m.end_time, reverse=True)[:10]:
                        try:
                            updated = await provider.market(old.market_id)
                            self.database.add_market(updated)
                            if updated.result_yes is not None:
                                logger.info("settlement_result", extra={"context": {"market_id": updated.market_id, "result_yes": updated.result_yes}})
                        except Exception as exc:
                            logger.warning("settlement_refresh_failed", extra={"context": {"market_id": old.market_id, "error": str(exc)}})
                    self._last_settlement_refresh = now
                return name, {"markets": len(markets), "books": books, "errors": failures}
            except Exception as exc:
                logger.error("provider_failure", extra={"context": {"provider": name, "error": str(exc)}})
                return name, {"error": str(exc)}

        return dict(await asyncio.gather(*(collect(name) for name in names)))

    async def backfill(self, source: str, start: datetime, end: datetime, step_seconds: int = 5) -> dict:
        if end <= start:
            raise ValueError("Backfill end must follow start")
        provider = self.provider(source)
        if source == "kalshi":
            observations = await provider.historical_markets(start, end)
            count = sum(self.database.add_market(market) for market in observations)
        else:
            if source == "pyth":
                observations = await provider.historical_prices(start, end, step_seconds=step_seconds)
            else:
                observations = await provider.historical_prices(start, end)
            count = sum(self.database.add_price(point, "spot" if source == "pyth" else "comex") for point in observations)
        return {"source": source, "observations": count, "availability": "receipt_time" if self.settings.historical_latency_seconds is None else "assumed_latency"}

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()
