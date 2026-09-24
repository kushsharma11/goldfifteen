"""Historical snapshots with official labels and point-in-time market metadata."""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import chain
from typing import Sequence

import pandas as pd

from gold_model.data.models import BookSnapshot, MarketWindow, PricePoint
from gold_model.features.builder import FEATURE_COLUMNS, FeatureBuilder, FeatureUnavailable

logger = logging.getLogger(__name__)
DATASET_COLUMNS = (
    "market_id", "timestamp", "market_start", "market_end", "label_available_at", "label",
    *FEATURE_COLUMNS,
)


def build_dataset(
    markets: Sequence[MarketWindow],
    spot: Sequence[PricePoint],
    comex: Sequence[PricePoint] = (),
    books: Sequence[BookSnapshot] = (),
    snapshot_seconds: int = 60,
    builder: FeatureBuilder | None = None,
) -> pd.DataFrame:
    """Sample strictly before expiry; attach only an official resolved outcome.

    All versions of market metadata should be supplied, including observations
    while open and after resolution. The label's availability time is retained
    so chronological training can purge outcomes not known by its cutoff.
    A final market response is never used retrospectively to supply a target.
    Missing core data produces a logged skip, exposed in ``frame.attrs['skipped']``.
    """
    if not isinstance(snapshot_seconds, int) or isinstance(snapshot_seconds, bool) or snapshot_seconds <= 0:
        raise ValueError("snapshot_seconds must be a positive integer")
    builder = builder or FeatureBuilder()
    ordered_spot = sorted(spot, key=lambda item: item.timestamp)
    ordered_comex = sorted(comex, key=lambda item: item.timestamp)
    spot_times = [point.timestamp for point in ordered_spot]
    comex_times = [point.timestamp for point in ordered_comex]
    market_books: dict[str, list[BookSnapshot]] = defaultdict(list)
    for book in books:
        market_books[book.market_id].append(book)
    versions: dict[str, list[MarketWindow]] = defaultdict(list)
    for market in markets:
        versions[market.market_id].append(market)
    rows: list[dict] = []
    skipped: list[dict] = []

    def skip(market_id: str, at: datetime | None, reason: str) -> None:
        record = {
            "market_id": market_id,
            "timestamp": at.isoformat() if at is not None else None,
            "reason": reason,
        }
        skipped.append(record)
        logger.info("dataset_snapshot_skipped", extra={"context": record})

    for market_id, history in versions.items():
        history.sort(key=lambda item: item.available_at)
        resolved = [market for market in history if market.result_yes is not None]
        if not resolved:
            skip(market_id, None, "No official market outcome is available")
            continue
        final = resolved[-1]
        label_available = max(final.available_at, final.settlement_time or final.end_time)
        # Scheduling uses earliest observed interval; all feature metadata below
        # uses the latest version available at each snapshot, never final fields.
        first = history[0]
        at = first.start_time + timedelta(seconds=snapshot_seconds)
        while at < first.end_time:
            known = [market for market in history if market.available_at <= at]
            if not known:
                skip(market_id, at, "No market metadata was available at prediction time")
            else:
                market = known[-1]
                try:
                    # Only the longest feature lookback is required. Availability
                    # remains checked by the builder, including late-arriving ticks.
                    lookback = at - timedelta(hours=1)
                    spot_slice = ordered_spot[
                        bisect_left(spot_times, lookback) : bisect_right(spot_times, at)
                    ]
                    comex_slice = ordered_comex[
                        bisect_left(comex_times, lookback) : bisect_right(comex_times, at)
                    ]
                    features = builder.build(at, market, spot_slice, comex_slice, market_books[market_id])
                except FeatureUnavailable as error:
                    skip(market_id, at, str(error))
                else:
                    rows.append({
                        "market_id": market_id,
                        "timestamp": at,
                        "market_start": market.start_time,
                        "market_end": market.end_time,
                        "label_available_at": label_available,
                        "label": int(final.result_yes),
                        **features,
                    })
            at += timedelta(seconds=snapshot_seconds)
    frame = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    for column in ("timestamp", "market_start", "market_end", "label_available_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if not frame.empty:
        frame = frame.sort_values(["timestamp", "market_id"]).reset_index(drop=True)
    frame.attrs["skipped"] = skipped
    frame.attrs["units"] = {
        "price": "USD per troy ounce", "return": "fraction", "volatility": "USD per sqrt second",
        "quote": "probability in [0, 1]", "expected_remaining_move": "USD per troy ounce",
    }
    assumptions: dict[tuple, dict] = {}
    for provider, metadata in chain(
        ((point.provider, point.metadata) for point in chain(spot, comex)),
        (("kalshi", market.raw.get("_availability", {})) for market in markets),
    ):
        if metadata.get("availability_basis") == "assumed_historical_latency":
            latency = metadata.get("historical_latency_seconds")
            assumption = metadata.get("assumption")
            item = {
                "provider": provider,
                "availability_basis": "assumed_historical_latency",
                "historical_latency_seconds": latency,
            }
            if assumption is not None:
                item["assumption"] = assumption
            assumptions[(provider, latency, assumption)] = item
    frame.attrs["availability_assumptions"] = list(assumptions.values())
    return frame
