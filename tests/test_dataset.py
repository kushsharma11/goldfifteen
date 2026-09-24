"""Label provenance and market revision leakage tests."""

from datetime import UTC, datetime, timedelta

import pytest

from gold_model.data.dataset import build_dataset
from gold_model.data.models import MarketWindow, PricePoint

START = datetime(2026, 1, 2, 14, tzinfo=UTC)
END = START + timedelta(minutes=15)


def market(**updates):
    return MarketWindow(
        **(
            {
                "market_id": "gold-1",
                "start_time": START,
                "end_time": END,
                "available_at": START,
                "target_price": 4300,
            }
            | updates
        )
    )


def prices():
    return [
        PricePoint(
            timestamp=START + timedelta(seconds=second),
            available_at=START + timedelta(seconds=second),
            price=4300 + 0.05 * (second // 5 % 3),
            provider="pyth",
        )
        for second in range(-3600, 901, 5)
    ]


def test_only_official_label_and_no_final_target_leakage():
    final = market(
        available_at=END + timedelta(seconds=20),
        target_price=9999,
        settled_price=1,
        result_yes=True,
        status="settled",
        settlement_time=END,
    )
    frame = build_dataset([market(), final], prices())
    assert len(frame) == 14
    assert frame["target_price"].eq(4300).all()
    assert frame["label"].eq(1).all()  # Official result wins; settled_price is not reconstructed.
    assert frame["label_available_at"].eq(END + timedelta(seconds=20)).all()
    assert frame["timestamp"].min() == START + timedelta(minutes=1)
    assert frame["timestamp"].max() == END - timedelta(minutes=1)
    assert frame["market_id"].nunique() == 1
    assert "settled_price" not in frame.columns


def test_missing_official_result_does_not_reconstruct_label():
    final = market(available_at=END, settled_price=4500, status="settled")
    frame = build_dataset([market(), final], prices())
    assert frame.empty
    assert "official market outcome" in frame.attrs["skipped"][0]["reason"]


def test_final_only_history_cannot_supply_past_market_metadata():
    final = market(available_at=END, result_yes=False, status="settled")
    frame = build_dataset([final], prices())
    assert frame.empty
    assert len(frame.attrs["skipped"]) == 14
    assert all("metadata" in item["reason"] for item in frame.attrs["skipped"])


def test_target_revision_is_used_only_after_availability():
    revised = market(available_at=START + timedelta(minutes=8), target_price=4301)
    final = market(available_at=END, result_yes=False, status="settled")
    frame = build_dataset([final, market(), revised], prices())
    assert frame.loc[frame["timestamp"] < revised.available_at, "target_price"].eq(4300).all()
    assert frame.loc[frame["timestamp"] >= revised.available_at, "target_price"].eq(4301).all()
    assert frame["label"].eq(0).all()


def test_late_downloaded_historical_ticks_are_unavailable():
    delayed = [
        point.model_copy(update={"available_at": END + timedelta(hours=1)}) for point in prices()
    ]
    final = market(available_at=END, result_yes=True, status="settled")
    frame = build_dataset([market(), final], delayed)
    assert frame.empty
    assert all("Pyth observation" in item["reason"] for item in frame.attrs["skipped"])


def test_configurable_snapshot_interval_and_empty_schema():
    final = market(available_at=END, result_yes=True, status="settled")
    frame = build_dataset([market(), final], prices(), snapshot_seconds=120)
    assert len(frame) == 7
    assert frame["seconds_remaining"].tolist() == [780, 660, 540, 420, 300, 180, 60]
    empty = build_dataset([], [])
    assert "label_available_at" in empty.columns
    assert "baseline_probability" in empty.columns
    assert empty.attrs["skipped"] == []


def test_retrospective_availability_assumptions_are_exposed_in_dataset_metadata():
    source_metadata = {
        "availability_basis": "assumed_historical_latency",
        "historical_latency_seconds": 2.0,
        "downloaded_at": (END + timedelta(days=1)).isoformat(),
    }
    assumed = [point.model_copy(update={"metadata": source_metadata}) for point in prices()]
    final = market(available_at=END, result_yes=True, status="settled")
    frame = build_dataset([market(), final], assumed)
    assert frame.attrs["availability_assumptions"] == [
        {
            "provider": "pyth",
            "availability_basis": "assumed_historical_latency",
            "historical_latency_seconds": 2.0,
        }
    ]


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_invalid_snapshot_intervals_fail(interval):
    with pytest.raises(ValueError, match="positive integer"):
        build_dataset([], [], snapshot_seconds=interval)
