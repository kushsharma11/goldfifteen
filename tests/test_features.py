"""Explicit point-in-time and quantitative feature invariants."""

from datetime import UTC, datetime, timedelta
from math import sqrt

import pytest

from gold_model.data.models import BookSnapshot, MarketWindow, PricePoint
from gold_model.features import FeatureBuilder, FeatureUnavailable
from gold_model.features.momentum import horizon_return
from gold_model.features.volatility import realized_volatility

AT = datetime(2026, 1, 2, 14, 10, tzinfo=UTC)


def prices(at=AT, horizon=3600, step=5, provider="pyth", contract=None):
    return [
        PricePoint(
            timestamp=at - timedelta(seconds=horizon - offset),
            available_at=at - timedelta(seconds=horizon - offset),
            price=4300 + 0.05 * (offset // step % 3),
            provider=provider,
            contract=contract,
        )
        for offset in range(0, horizon + 1, step)
    ]


def market(**updates):
    payload = {
        "market_id": "gold-15m",
        "start_time": AT - timedelta(minutes=10),
        "end_time": AT + timedelta(minutes=5),
        "available_at": AT - timedelta(minutes=10),
        "target_price": 4300,
    }
    return MarketWindow(**(payload | updates))


def book(at=AT, **updates):
    payload = {
        "market_id": "gold-15m",
        "timestamp": at,
        "available_at": at,
        "yes_bid": 0.48,
        "yes_ask": 0.52,
        "no_bid": 0.48,
        "no_ask": 0.52,
        "yes_ask_size": 12,
        "no_ask_size": 18,
        "yes_bids": [[0.48, 20]],
        "no_bids": [[0.48, 10]],
    }
    return BookSnapshot(**(payload | updates))


def test_core_units_z_score_and_book_features():
    features = FeatureBuilder().build(AT, market(), prices(), books=[book()])
    assert features["seconds_remaining"] == 300
    assert features["minutes_remaining"] == 5
    assert features["expected_remaining_move"] == pytest.approx(
        features["volatility_5m"] * sqrt(300)
    )
    assert features["target_z_score"] == pytest.approx(
        features["distance_from_target"] / features["expected_remaining_move"]
    )
    assert features["kalshi_mid_probability"] == 0.5
    assert features["kalshi_spread"] == pytest.approx(0.04)
    assert features["kalshi_order_book_imbalance"] == pytest.approx(1 / 3)
    assert features["kalshi_yes_ask_size"] == 12
    assert 0 < features["baseline_probability"] < 1


def test_future_and_late_published_observations_never_affect_features():
    builder = FeatureBuilder()
    original = builder.build(AT, market(), prices(), books=[book()])
    future = PricePoint(
        timestamp=AT + timedelta(seconds=1),
        available_at=AT + timedelta(seconds=1),
        price=9000,
        provider="pyth",
    )
    late_revision = PricePoint(
        timestamp=AT,
        available_at=AT + timedelta(seconds=2),
        price=8500,
        provider="pyth",
    )
    late_book = book(yes_bid=0.8, yes_ask=0.9, available_at=AT + timedelta(seconds=1))
    rebuilt = builder.build(
        AT,
        market(settled_price=9999, result_yes=False),
        prices() + [future, late_revision],
        books=[book(), late_book, book(AT + timedelta(seconds=1))],
    )
    assert original == rebuilt


def test_latest_known_revision_and_backward_asof():
    revised = PricePoint(
        timestamp=AT - timedelta(seconds=5), available_at=AT, price=4301, provider="pyth"
    )
    points = prices(at=AT - timedelta(seconds=5))
    features = FeatureBuilder().build(AT, market(), points + [revised])
    assert features["current_gold_price"] == 4301
    assert features["spot_age_seconds"] == 5


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"target_price": None}, "target"),
        ({"available_at": AT + timedelta(seconds=1)}, "not yet available"),
        ({"end_time": AT}, "expired"),
        ({"status": "settled"}, "not open"),
        ({"start_time": AT + timedelta(seconds=1)}, "not started"),
    ],
)
def test_unavailable_market_is_rejected(updates, reason):
    with pytest.raises(FeatureUnavailable, match=reason):
        FeatureBuilder().build(AT, market(**updates), prices())


def test_missing_stale_and_insufficient_spot_are_rejected():
    builder = FeatureBuilder()
    with pytest.raises(FeatureUnavailable, match="No Pyth"):
        builder.build(AT, market(), [])
    with pytest.raises(FeatureUnavailable, match="stale"):
        builder.build(AT, market(), prices(at=AT - timedelta(seconds=16)))
    with pytest.raises(FeatureUnavailable, match="Insufficient"):
        builder.build(AT, market(), prices(horizon=120))
    with pytest.raises(ValueError, match="timezone-aware"):
        builder.build(AT.replace(tzinfo=None), market(), prices())


def test_flat_price_volatility_has_finite_floor_and_half_probability_at_target():
    flat = [point.model_copy(update={"price": 4300.0}) for point in prices()]
    features = FeatureBuilder().build(AT, market(), flat)
    assert features["volatility_5m"] == 0
    assert features["expected_remaining_move"] > 0
    assert features["target_z_score"] == 0
    assert features["baseline_probability"] == 0.5


def test_cadence_and_gap_rules_do_not_manufacture_short_returns():
    minute_prices = prices(step=60)
    assert horizon_return(minute_prices, AT, 10) is None
    assert horizon_return(minute_prices, AT, 30) is None
    assert horizon_return(minute_prices, AT, 60) is not None
    assert horizon_return(minute_prices, AT, 300) is not None
    assert realized_volatility(minute_prices, AT, 300) is not None
    gap = [
        point
        for point in prices(horizon=300)
        if not AT - timedelta(seconds=200) < point.timestamp < AT - timedelta(seconds=100)
    ]
    assert realized_volatility(gap, AT, 300) is None
    assert horizon_return([minute_prices[-2]], AT, 60) is None


def test_dollar_volatility_matches_quadratic_variation():
    sample = prices(horizon=300, step=60)
    expected = sqrt(
        sum((b.price - a.price) ** 2 for a, b in zip(sample, sample[1:], strict=False)) / 300
    )
    assert realized_volatility(sample, AT, 300) == pytest.approx(expected)


@pytest.mark.parametrize(
    "comex",
    [
        [],
        prices(at=AT - timedelta(seconds=31), provider="comex", contract="GCZ6"),
        [
            point.model_copy(update={"delayed": True})
            for point in prices(provider="comex", contract="GCZ6")
        ],
    ],
)
def test_unavailable_comex_is_missing_instead_of_zero(comex):
    features = FeatureBuilder().build(AT, market(), prices(), comex)
    assert all(
        features[f"comex_return_{horizon}"] is None for horizon in ("10s", "30s", "1m", "3m", "5m")
    )
    assert features["current_comex_price"] is None


def test_comex_returns_never_cross_contract_roll():
    futures = prices(provider="comex", contract="GCZ6")
    futures[-1] = futures[-1].model_copy(update={"contract": "GCG7", "price": 4400.0})
    features = FeatureBuilder().build(AT, market(), prices(), futures)
    assert features["comex_return_10s"] is None
    assert features["comex_return_5m"] is None
    good = FeatureBuilder().build(AT, market(), prices(), prices(provider="comex", contract="GCZ6"))
    assert good["comex_return_10s"] is not None


def test_future_and_late_comex_ticks_cannot_change_returns():
    futures = prices(provider="comex", contract="GCZ6")
    late = futures[-1].model_copy(
        update={"price": 4900.0, "available_at": AT + timedelta(seconds=1)}
    )
    future = late.model_copy(update={"timestamp": AT + timedelta(seconds=1)})
    builder = FeatureBuilder()
    assert builder.build(AT, market(), prices(), futures) == builder.build(
        AT, market(), prices(), futures + [late, future]
    )


@pytest.mark.parametrize("identity", [{"feed_id": "new-feed"}, {"provider": "other-spot"}])
def test_spot_feed_or_provider_change_requires_its_own_volatility_warmup(identity):
    history = prices()
    switched = history[-1].model_copy(update={"price": 9999.0, **identity})
    with pytest.raises(FeatureUnavailable, match="Insufficient"):
        FeatureBuilder().build(AT, market(), history + [switched])
    new_history = [point.model_copy(update=identity) for point in prices(horizon=300)]
    features = FeatureBuilder().build(AT, market(), history + new_history)
    assert features["volatility_5m"] is not None
    assert features["volatility_15m"] is None
    assert features["volatility_60m"] is None


def test_comex_provider_change_does_not_create_cross_provider_returns():
    history = prices(provider="first-comex", contract="GCZ6")
    switched = history[-1].model_copy(update={"provider": "second-comex", "price": 4999.0})
    features = FeatureBuilder().build(AT, market(), prices(), history + [switched])
    assert features["comex_return_10s"] is None
    assert features["comex_return_5m"] is None


def test_stale_book_retains_age_but_no_executable_quotes():
    features = FeatureBuilder().build(
        AT, market(), prices(), books=[book(AT - timedelta(seconds=16))]
    )
    assert features["book_age_seconds"] == 16
    assert features["kalshi_yes_ask"] is None
    assert features["kalshi_mid_probability"] is None


def test_books_of_other_markets_are_not_joined():
    features = FeatureBuilder().build(AT, market(), prices(), books=[book(market_id="other")])
    assert features["kalshi_yes_ask"] is None
