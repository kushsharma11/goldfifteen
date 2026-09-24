"""Live decision guards and persisted research predictions; no network or orders."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from gold_model.config import Settings
from gold_model.data.database import Database, FeatureObservation, ModelPrediction
from gold_model.data.models import BookSnapshot, MarketWindow, PricePoint
from gold_model.features import FeatureUnavailable
from gold_model.models.logistic import LogisticModel
from gold_model.services.predictor import Predictor

AT = datetime(2026, 1, 2, 14, 10, tzinfo=timezone.utc)


@pytest.fixture
def database():
    database = Database("sqlite:///:memory:")
    database.init()
    database.add_market(MarketWindow(
        market_id="gold-live", start_time=AT - timedelta(minutes=10),
        end_time=AT + timedelta(minutes=5), available_at=AT - timedelta(minutes=10),
        target_price=4300,
    ))
    for offset in range(0, 301, 5):
        timestamp = AT - timedelta(seconds=300 - offset)
        database.add_price(PricePoint(
            timestamp=timestamp, available_at=timestamp, price=4300 + 0.05 * (offset // 5 % 3),
            provider="pyth",
        ))
    yield database
    database.close()


def settings(**updates):
    return Settings(_env_file=None, **({
        "market_reference_verified": True, "include_fees": False,
        "sizing_method": "fixed_contracts", "fixed_contracts": 3,
    } | updates))


def add_book(database, at=AT, depth=10):
    database.add_book(BookSnapshot(
        timestamp=at, available_at=at, market_id="gold-live",
        yes_bid=0.18, yes_ask=0.20, no_bid=0.80, no_ask=0.82,
        yes_ask_size=depth, no_ask_size=depth,
    ))


def persisted(database, model):
    with Session(database.engine) as session:
        return [row.payload for row in session.scalars(select(model))]


def test_baseline_prediction_and_features_are_persisted(database):
    add_book(database)
    prediction = Predictor(settings(), database).predict(at=AT)
    assert prediction.record.action == "YES"
    assert prediction.record.probability_yes == pytest.approx(0.5)
    assert prediction.record.probability_yes + prediction.record.probability_no == 1
    assert prediction.record.recommended_position == 3
    assert prediction.record.risk_dollars == pytest.approx(0.60)
    assert prediction.record.model_version == "normal-baseline-v1"
    assert persisted(database, ModelPrediction) == [prediction.record.model_dump(mode="json")]
    stored_features = persisted(database, FeatureObservation)
    assert len(stored_features) == 1
    assert stored_features[0]["features"] == prediction.features
    assert stored_features[0]["market_id"] == "gold-live"


def test_stale_spot_refuses_prediction_and_does_not_persist_signal(database):
    with pytest.raises(FeatureUnavailable, match="stale"):
        Predictor(settings(), database).predict(at=AT + timedelta(seconds=16))
    assert persisted(database, ModelPrediction) == []


def test_missing_book_keeps_probability_but_passes_with_zero_position(database):
    prediction = Predictor(settings(), database).predict(at=AT)
    assert 0 < prediction.record.probability_yes < 1
    assert prediction.record.action == "PASS"
    assert prediction.record.recommended_position == 0
    assert prediction.record.risk_dollars == 0
    assert prediction.record.kalshi_yes_ask is None
    assert prediction.record.yes_edge is None


def test_stale_book_is_never_treated_as_executable(database):
    add_book(database, AT - timedelta(seconds=16))
    prediction = Predictor(settings(), database).predict(at=AT)
    assert prediction.features["book_age_seconds"] == 16
    assert prediction.record.action == "PASS"
    assert prediction.record.kalshi_yes_ask is None
    assert prediction.record.recommended_position == 0


def test_unverified_settlement_reference_requires_pass_despite_positive_edge(database):
    add_book(database)
    prediction = Predictor(settings(market_reference_verified=False), database).predict(at=AT)
    assert prediction.record.yes_edge > 0.2
    assert prediction.record.action == "PASS"
    assert prediction.record.recommended_position == 0
    assert "reference/feed parity" in prediction.record.reason


def test_expired_market_never_produces_an_actionable_prediction(database):
    with pytest.raises(FeatureUnavailable, match="no currently open"):
        Predictor(settings(), database).predict(at=AT + timedelta(minutes=5))
    assert persisted(database, ModelPrediction) == []


def test_unknown_depth_requires_pass_and_known_depth_limits_position(database):
    add_book(database, depth=None)
    no_depth = Predictor(settings(), database).predict(at=AT)
    assert no_depth.record.action == "PASS"
    assert no_depth.record.recommended_position == 0
    assert "depth" in no_depth.record.reason
    add_book(database, depth=2)
    limited = Predictor(settings(), database).predict(at=AT)
    assert limited.record.action == "YES"
    assert limited.record.recommended_position == 2


def test_live_model_requiring_missing_comex_refuses_without_zero_substitution(database):
    # Independent synthetic training data is used only to exercise a fitted
    # model's input contract; these observations are not a profitability sample.
    rows = []
    for index in range(6):
        start = AT - timedelta(hours=2) + timedelta(minutes=15 * index)
        end = start + timedelta(minutes=15)
        rows.append({
            "market_id": f"training-{index}", "timestamp": start + timedelta(minutes=5),
            "market_start": start, "market_end": end, "label_available_at": end,
            "label": index % 2, "comex_return_1m": 0.001 * (2 * (index % 2) - 1),
        })
    frame = pd.DataFrame(rows)
    model = LogisticModel(feature_names=["comex_return_1m"]).fit(frame.iloc[:4], frame.iloc[4:])
    add_book(database)
    with pytest.raises(ValueError, match="comex_return_1m.*no zero substitution"):
        Predictor(settings(), database, model).predict(at=AT)
    assert persisted(database, ModelPrediction) == []


def test_future_market_revision_and_late_tick_do_not_enter_live_prediction(database):
    add_book(database)
    original = Predictor(settings(), database).predict(at=AT)
    database.add_price(PricePoint(
        timestamp=AT, available_at=AT + timedelta(seconds=1), price=9999, provider="pyth",
    ))
    database.add_market(MarketWindow(
        market_id="gold-live", start_time=AT - timedelta(minutes=10), end_time=AT + timedelta(minutes=5),
        available_at=AT + timedelta(seconds=1), target_price=9999, status="closed",
    ))
    repeat = Predictor(settings(), database).predict(at=AT)
    assert repeat.features == original.features
    assert repeat.record == original.record
