"""Price momentum in fractional returns, using observed backward/as-of samples."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence

from gold_model.data.models import PricePoint


def available_prices(points: Sequence[PricePoint], at: datetime) -> list[PricePoint]:
    """Select known observations and the latest known revision at each event time.

    Event time alone is insufficient: an old tick downloaded later was not known
    to a historical prediction. Both clocks must be on or before ``at``.
    """
    by_timestamp: dict[datetime, PricePoint] = {}
    for point in points:
        if point.timestamp <= at and point.available_at <= at:
            prior = by_timestamp.get(point.timestamp)
            if prior is None or prior.available_at <= point.available_at:
                by_timestamp[point.timestamp] = point
    return sorted(by_timestamp.values(), key=lambda point: point.timestamp)


def horizon_return(
    points: Sequence[PricePoint],
    at: datetime,
    seconds: int,
    *,
    require_same_contract: bool = False,
) -> float | None:
    """Return ``latest / asof(at - horizon) - 1`` only at supported cadence.

    Endpoints may lag by at most a quarter of the requested horizon. No gap
    inside the interval may exceed the horizon. These rules reject a claimed
    ten-second return derived from one-minute observations without filling or
    interpolating missing prices. COMEX returns never cross a contract change.
    """
    if seconds <= 0:
        raise ValueError("Return horizon must be positive")
    known = available_prices(points, at)
    if not known:
        return None
    cutoff = at - timedelta(seconds=seconds)
    anchors = [index for index, point in enumerate(known) if point.timestamp <= cutoff]
    if not anchors:
        return None
    interval = known[anchors[-1] :]
    if len(interval) < 2:
        return None
    first, last = interval[0], interval[-1]
    endpoint_tolerance = seconds / 4
    if (
        (cutoff - first.timestamp).total_seconds() > endpoint_tolerance
        or (at - last.timestamp).total_seconds() > endpoint_tolerance
    ):
        return None
    gaps = [
        (right.timestamp - left.timestamp).total_seconds()
        for left, right in zip(interval, interval[1:])
    ]
    if max(gaps) > seconds:
        return None
    if require_same_contract and (
        last.contract is None or any(point.contract != last.contract for point in interval)
    ):
        return None
    return float(last.price / first.price - 1.0)
