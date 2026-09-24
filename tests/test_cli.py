"""Offline end-to-end smoke tests. Fixtures are not historical market evidence."""

import json
import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from typer.testing import CliRunner

from gold_model.cli.main import app
from gold_model.data.database import Database
from gold_model.data.models import BookSnapshot, MarketWindow, PricePoint

runner = CliRunner()


@pytest.fixture
def research_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    url = f"sqlite:///{tmp_path / 'observations.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("DATASET_PATH", str(tmp_path / "dataset.csv"))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "logistic.joblib"))
    monkeypatch.setenv("REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("SNAPSHOT_SECONDS", "300")
    monkeypatch.setenv("MARKET_REFERENCE_VERIFIED", "true")
    db = Database(url)
    db.init()
    start = datetime(2026, 1, 5, 12, tzinfo=UTC)
    # Tests deliberately use labeled fixtures. They never reach a production
    # database or appear in reported market-performance examples.
    for seconds in range(-360, 18 * 900 + 1, 10):
        at = start + timedelta(seconds=seconds)
        price = 4300 + math.sin(seconds / 100) * 3 + math.cos(seconds / 37) * 0.2
        db.add_price(
            PricePoint(timestamp=at, available_at=at, price=price, provider="test_fixture")
        )
    for i in range(18):
        opened = start + timedelta(minutes=15 * i)
        closed = opened + timedelta(minutes=15)
        market = MarketWindow(
            market_id=f"TEST-{i}",
            start_time=opened,
            end_time=closed,
            available_at=opened,
            target_price=4300,
            status="open",
        )
        db.add_market(market)
        db.add_market(
            market.model_copy(
                update={
                    "available_at": closed + timedelta(seconds=1),
                    "settlement_time": closed,
                    "result_yes": bool(i % 2),
                    "status": "settled",
                }
            )
        )
        for offset in (300, 600):
            at = opened + timedelta(seconds=offset)
            db.add_book(
                BookSnapshot(
                    timestamp=at,
                    available_at=at,
                    market_id=market.market_id,
                    yes_bid=0.44,
                    yes_ask=0.46,
                    no_bid=0.54,
                    no_ask=0.56,
                    yes_ask_size=50,
                    no_ask_size=50,
                )
            )
    db.close()
    return tmp_path


def invoke_ok(args):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, f"{args}: {result.output}\n{result.exception!r}"
    return result


def test_full_offline_research_workflow(research_workspace):
    root = research_workspace
    invoke_ok(["init-db"])
    invoke_ok(["build-dataset"])
    dataset = pd.read_csv(root / "dataset.csv")
    assert len(dataset) == 36
    invoke_ok(["train", "baseline"])
    invoke_ok(["train", "logistic"])
    assert (root / "logistic.joblib.json").exists()
    invoke_ok(["evaluate"])
    report = json.loads((root / "reports/evaluation.json").read_text())
    assert "common_support" in report["probability_comparison"]
    invoke_ok(["backtest"])
    report = json.loads((root / "reports/backtest.json").read_text())
    assert report["metrics"]["mode"] == "executable"
    assert len(report["thresholds"]) == 7
    assert len(pd.read_csv(root / "reports/oos_predictions.csv")) > 0
    invoke_ok(
        [
            "backtest",
            "--walk-forward",
            "--min-train-markets",
            "6",
            "--calibration-markets",
            "4",
            "--test-markets",
            "4",
        ]
    )
    report = json.loads((root / "reports/backtest.json").read_text())
    assert report["evaluation"] == "walk-forward out-of-sample"
    assert len(report["folds"]) == 2


def test_help_and_empty_data_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    invoke_ok(["--help"])
    invoke_ok(["init-db"])
    result = runner.invoke(app, ["build-dataset"])
    assert result.exit_code == 1
    assert "No eligible snapshots" in result.output
    result = runner.invoke(app, ["predict"])
    assert result.exit_code == 1
    assert "NO SIGNAL" in result.output


def test_dataset_tampering_is_detected(research_workspace):
    invoke_ok(["build-dataset"])
    path = research_workspace / "dataset.csv"
    path.write_text(path.read_text().replace("4300.0", "4301.0", 1))
    result = runner.invoke(app, ["train", "logistic"])
    assert result.exit_code == 1
    assert "hash differs" in result.output


def test_naive_backfill_date_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["backfill", "kalshi", "--start", "2026-01-01", "--end", "2026-01-02"]
    )
    assert result.exit_code == 1
    assert "Naive datetime" in result.output
