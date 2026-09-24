"""Reusable, point-in-time feature generation for research and monitoring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import erf, isfinite, sqrt

from gold_model.data.models import BookSnapshot, MarketWindow, PricePoint
from gold_model.features.momentum import available_prices, horizon_return
from gold_model.features.volatility import realized_volatility

RETURN_HORIZONS = {"10s": 10, "30s": 30, "1m": 60, "3m": 180, "5m": 300}
VOLATILITY_HORIZONS = {"5m": 300, "15m": 900, "60m": 3600}
CORE_FEATURES = (
    "current_gold_price",
    "target_price",
    "distance_from_target",
    "distance_from_target_pct",
    "seconds_remaining",
    "minutes_remaining",
    "spot_age_seconds",
    "expected_remaining_move",
    "target_z_score",
    "baseline_probability",
)
BOOK_FEATURES = (
    "kalshi_yes_bid",
    "kalshi_yes_ask",
    "kalshi_no_bid",
    "kalshi_no_ask",
    "kalshi_yes_ask_size",
    "kalshi_no_ask_size",
    "yes_ask_size",
    "no_ask_size",
    "kalshi_mid_probability",
    "kalshi_spread",
    "kalshi_order_book_imbalance",
    "book_age_seconds",
)
FEATURE_COLUMNS = (
    *CORE_FEATURES,
    *(f"gold_return_{name}" for name in RETURN_HORIZONS),
    *(f"volatility_{name}" for name in VOLATILITY_HORIZONS),
    *(f"comex_return_{name}" for name in RETURN_HORIZONS),
    "current_comex_price",
    "comex_age_seconds",
    *BOOK_FEATURES,
)


class FeatureUnavailable(ValueError):
    """Core information was not available for a valid prediction at this time."""


def _utc(at: datetime) -> datetime:
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("Feature timestamps must be timezone-aware")
    return at.astimezone(UTC)


def _depth_size(levels: list) -> float:
    total = 0.0
    for level in levels:
        if isinstance(level, dict):
            size = level.get("size", level.get("quantity", 0))
        else:
            size = level[1]
        total += float(size)
    return total


@dataclass(frozen=True)
class FeatureBuilder:
    """Build numeric features; optional feeds stay missing when unusable.

    Returns are fractions, quotes are probabilities in [0, 1], prices and
    expected remaining moves are USD per troy ounce, and volatility is
    USD/√second. ``min_volatility`` floors diffusion volatility in those units.
    The baseline assumes a driftless normal arithmetic price diffusion.
    """

    spot_max_age_seconds: float = 15.0
    book_max_age_seconds: float = 15.0
    comex_max_age_seconds: float = 30.0
    min_volatility: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "spot_max_age_seconds",
            "book_max_age_seconds",
            "comex_max_age_seconds",
            "min_volatility",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def build(
        self,
        at: datetime,
        market: MarketWindow,
        spot: Sequence[PricePoint],
        comex: Sequence[PricePoint] = (),
        books: Sequence[BookSnapshot] = (),
    ) -> dict[str, float | None]:
        at = _utc(at)
        if market.available_at > at:
            raise FeatureUnavailable("Market metadata was not yet available")
        if at < market.start_time:
            raise FeatureUnavailable("Market interval has not started")
        remaining = (market.end_time - at).total_seconds()
        if remaining <= 0:
            raise FeatureUnavailable("Market interval has expired")
        if market.target_price is None:
            raise FeatureUnavailable("Market target price is unknown")
        if market.status.lower() not in {"open", "active"}:
            raise FeatureUnavailable("Market is not open at prediction time")
        known_spot = available_prices(spot, at)
        if not known_spot:
            raise FeatureUnavailable("No Pyth observation was available")
        latest = known_spot[-1]
        spot_age = (at - latest.timestamp).total_seconds()
        if spot_age > self.spot_max_age_seconds or latest.delayed:
            raise FeatureUnavailable("Pyth price is stale or delayed")
        # A configured feed change must warm up its own history. Different
        # references/providers can have a level basis that creates fake returns.
        known_spot = [
            point
            for point in known_spot
            if (point.provider, point.feed_id) == (latest.provider, latest.feed_id)
        ]
        features: dict[str, float | None] = dict.fromkeys(FEATURE_COLUMNS)
        price, target = float(latest.price), float(market.target_price)
        features.update(
            current_gold_price=price,
            target_price=target,
            distance_from_target=price - target,
            distance_from_target_pct=(price - target) / target,
            seconds_remaining=remaining,
            minutes_remaining=remaining / 60,
            spot_age_seconds=spot_age,
        )
        for name, seconds in RETURN_HORIZONS.items():
            features[f"gold_return_{name}"] = horizon_return(known_spot, at, seconds)
        for name, seconds in VOLATILITY_HORIZONS.items():
            features[f"volatility_{name}"] = realized_volatility(known_spot, at, seconds)
        sigma = next(
            (
                features[f"volatility_{name}"]
                for name in VOLATILITY_HORIZONS
                if features[f"volatility_{name}"] is not None
            ),
            None,
        )
        if sigma is None:
            raise FeatureUnavailable("Insufficient continuous price history for volatility")
        expected_move = max(sigma, self.min_volatility) * sqrt(remaining)
        z_score = (price - target) / expected_move
        features.update(
            expected_remaining_move=expected_move,
            target_z_score=z_score,
            baseline_probability=max(1e-6, min(1 - 1e-6, 0.5 * (1 + erf(z_score / sqrt(2))))),
        )
        known_comex = available_prices(comex, at)
        if known_comex:
            latest_comex = known_comex[-1]
            age = (at - latest_comex.timestamp).total_seconds()
            if not latest_comex.delayed and age <= self.comex_max_age_seconds:
                real_time = [
                    point
                    for point in known_comex
                    if not point.delayed and point.provider == latest_comex.provider
                ]
                features["current_comex_price"] = float(latest_comex.price)
                features["comex_age_seconds"] = age
                for name, seconds in RETURN_HORIZONS.items():
                    features[f"comex_return_{name}"] = horizon_return(
                        real_time, at, seconds, require_same_contract=True
                    )
        self._book_features(features, books, market.market_id, at)
        return features

    def _book_features(
        self,
        features: dict[str, float | None],
        books: Sequence[BookSnapshot],
        market_id: str,
        at: datetime,
    ) -> None:
        eligible = [
            book
            for book in books
            if book.market_id == market_id and book.timestamp <= at and book.available_at <= at
        ]
        if not eligible:
            return
        # For identical clocks, the last supplied immutable revision wins,
        # matching price and market-version selection.
        book = max(reversed(eligible), key=lambda item: (item.timestamp, item.available_at))
        age = (at - book.timestamp).total_seconds()
        features["book_age_seconds"] = age
        if age > self.book_max_age_seconds:
            return
        for name in ("yes_bid", "yes_ask", "no_bid", "no_ask", "yes_ask_size", "no_ask_size"):
            value = getattr(book, name)
            features[f"kalshi_{name}"] = None if value is None else float(value)
        features["yes_ask_size"] = features["kalshi_yes_ask_size"]
        features["no_ask_size"] = features["kalshi_no_ask_size"]
        if book.yes_bid is not None and book.yes_ask is not None:
            features["kalshi_mid_probability"] = (book.yes_bid + book.yes_ask) / 2
            features["kalshi_spread"] = book.yes_ask - book.yes_bid
        if book.yes_bids and book.no_bids:
            yes_depth, no_depth = _depth_size(book.yes_bids), _depth_size(book.no_bids)
            if yes_depth + no_depth > 0:
                features["kalshi_order_book_imbalance"] = (yes_depth - no_depth) / (
                    yes_depth + no_depth
                )
