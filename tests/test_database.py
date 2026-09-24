from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from gold_model.config import Settings
from gold_model.data.database import Database, SpotObservation
from gold_model.data.models import MarketWindow, PricePoint
from gold_model.utils.time import as_utc, seconds_remaining


def test_utc_roundtrip_and_immutable_dedup(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'test.db'}")
    db.init()
    local = datetime(2026, 1, 1, 10, tzinfo=timezone(timedelta(hours=-5)))
    p = PricePoint(timestamp=local, available_at=local, price=4300, provider="test")
    assert db.add_price(p)
    assert not db.add_price(p)
    assert db.prices()[0].timestamp == datetime(2026, 1, 1, 15, tzinfo=UTC)
    with Session(db.engine) as session:
        assert session.scalar(select(SpotObservation)).timestamp.tzinfo is UTC
    with pytest.raises(ValidationError):
        p.price = 0
    db.close()


def test_market_revisions_are_kept(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'test.db'}")
    db.init()
    t = datetime(2026, 1, 1, tzinfo=UTC)
    market = MarketWindow(
        market_id="test",
        start_time=t,
        end_time=t + timedelta(minutes=15),
        available_at=t,
        target_price=4300,
    )
    db.add_market(market)
    db.add_market(
        market.model_copy(
            update={
                "result_yes": True,
                "available_at": t + timedelta(minutes=16),
                "status": "settled",
            }
        )
    )
    assert len(db.markets()) == 2
    assert db.markets()[0].result_yes is None


def test_naive_timestamps_and_bad_availability_rejected():
    with pytest.raises(ValueError, match="Naive"):
        as_utc(datetime(2026, 1, 1))
    t = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError):
        PricePoint(timestamp=t, available_at=t - timedelta(seconds=1), price=4300, provider="test")
    assert seconds_remaining(t, t + timedelta(seconds=1)) == 0


def test_configuration_bounds():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, min_edge=-1)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, kelly_fraction=2)
    assert Settings(_env_file=None).kelly_fraction == 0.25
