"""Realized gold volatility with explicit price/time units."""

from __future__ import annotations

from datetime import datetime, timedelta
from math import sqrt
from statistics import median
from typing import Sequence

from gold_model.data.models import PricePoint
from gold_model.features.momentum import available_prices


def realized_volatility(
    points: Sequence[PricePoint],
    at: datetime,
    seconds: int,
    *,
    min_increments: int = 5,
) -> float | None:
    """Estimate USD/√second as ``sqrt(sum(delta_price**2) / elapsed)``.

    This arithmetic diffusion estimate matches the baseline's dollar distance
    from target. It is not annualized or a percentage volatility. Require at
    least five increments, 90% horizon coverage, and no gaps larger than three
    normal sampling intervals or one fifth of the horizon. An incomplete
    warm-up window is missing, rather than misrepresented as a full window.
    """
    if seconds <= 0 or min_increments < 2:
        raise ValueError("Volatility requires a positive horizon and at least two increments")
    cutoff = at - timedelta(seconds=seconds)
    sample = [point for point in available_prices(points, at) if point.timestamp >= cutoff]
    if len(sample) < min_increments + 1:
        return None
    gaps = [
        (right.timestamp - left.timestamp).total_seconds()
        for left, right in zip(sample, sample[1:])
    ]
    elapsed = (sample[-1].timestamp - sample[0].timestamp).total_seconds()
    typical_gap = median(gaps)
    if (
        elapsed < 0.9 * seconds
        or (at - sample[-1].timestamp).total_seconds() > min(0.1 * seconds, typical_gap)
        or max(gaps) > min(seconds / 5, 3 * typical_gap)
    ):
        return None
    quadratic_variation = sum(
        (right.price - left.price) ** 2 for left, right in zip(sample, sample[1:])
    )
    return float(sqrt(quadratic_variation / elapsed))
