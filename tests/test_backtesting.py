from dataclasses import replace

import pandas as pd
import pytest

from gold_model.backtesting.engine import BacktestConfig, run_backtest, threshold_sweep
from gold_model.trading.edge import CostConfig
from gold_model.trading.sizing import SizingConfig


def row(market_id="a", minute=0, probability=0.8, label=1, settlement_delay=0):
    opening = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(minutes=minute)
    return {
        "market_id": market_id,
        "timestamp": opening + pd.Timedelta(minutes=1),
        "market_start": opening,
        "market_end": opening + pd.Timedelta(minutes=15),
        "label_available_at": opening + pd.Timedelta(minutes=15 + settlement_delay),
        "label": label,
        "probability_yes": probability,
        "baseline_probability": 0.6,
        "kalshi_mid_probability": 0.5,
        "kalshi_yes_ask": 0.6,
        "kalshi_no_ask": 0.4,
        "yes_ask_size": 100,
        "no_ask_size": 100,
        "book_age_seconds": 1,
        "volatility_5m": 0.1,
    }


def config(**kwargs):
    return BacktestConfig(
        initial_bankroll=100,
        costs=CostConfig(include_fees=False),
        sizing=SizingConfig(
            method="fixed_contracts",
            fixed_contracts=10,
            max_position_dollars=100,
            max_fraction_of_bankroll_per_market=1,
        ),
        **kwargs,
    )


def test_yes_and_no_payoffs_at_asks_with_one_trade_per_market():
    yes, no = row(), row("b", 15, probability=0.2, label=0)
    later = dict(yes, timestamp=yes["timestamp"] + pd.Timedelta(minutes=2))
    result = run_backtest(pd.DataFrame([no, later, yes]), config())
    assert result.metrics["trades"] == 2
    assert result.trades.side.tolist() == ["YES", "NO"]
    assert result.trades.pnl.tolist() == pytest.approx([4, 6])
    assert result.metrics["cumulative_pnl"] == pytest.approx(10)
    assert result.metrics["roi"] == pytest.approx(1)
    assert result.metrics["pass_count"] == 1
    assert result.metrics["final_bankroll"] == pytest.approx(110)


def test_cash_reserved_until_outcome_available_not_market_close():
    first = row("a", 0, settlement_delay=20)
    second = row("b", 15)
    cfg = replace(config(), initial_bankroll=6)
    result = run_backtest(pd.DataFrame([first, second]), cfg)
    assert result.metrics["trades"] == 1
    assert result.decisions.iloc[1].available_cash_before == pytest.approx(0)
    assert result.metrics["final_bankroll"] == 10


def test_missing_or_stale_depth_never_silently_uses_midpoint():
    quote = row()
    quote["yes_ask_size"] = None
    result = run_backtest(pd.DataFrame([quote]), config())
    assert result.metrics["trades"] == 0
    idealized = run_backtest(pd.DataFrame([quote]), config(mode="idealized"))
    assert idealized.metrics["trades"] == 1
    assert idealized.trades.execution_price.iloc[0] == 0.5
    assert "IDEALIZED" in idealized.metrics["execution_assumption"]
    stale = dict(row(), book_age_seconds=100)
    assert run_backtest(pd.DataFrame([stale]), config()).metrics["trades"] == 0


def test_loss_drawdown_depth_and_order_level_fees():
    quote = dict(row(label=0), yes_ask_size=2)
    cfg = replace(config(), costs=CostConfig(fee_per_contract=0.02, slippage_cents=1))
    result = run_backtest(pd.DataFrame([quote]), cfg)
    assert result.trades.contracts.iloc[0] == 2
    assert result.trades.risk_dollars.iloc[0] == pytest.approx(1.26)
    assert result.metrics["cumulative_pnl"] == pytest.approx(-1.26)
    assert result.metrics["max_drawdown"] == pytest.approx(1.26)
    assert result.metrics["fees_included"] is True
    assert result.metrics["sharpe_like"] is None


def test_threshold_sensitivity_reports_fixed_oos_predictions():
    sweep = threshold_sweep(pd.DataFrame([row()]), thresholds=[0.1, 0.3], config=config())
    assert sweep.trades.tolist() == [1, 0]
    assert sweep.cumulative_pnl.tolist() == [4, 0]


def test_future_book_timestamps_and_invalid_labels_rejected():
    bad = dict(row(), book_age_seconds=-1)
    assert run_backtest(pd.DataFrame([bad]), config()).metrics["trades"] == 0
    bad = dict(row(), label_available_at=pd.Timestamp("2025-01-01", tz="UTC"))
    with pytest.raises(ValueError, match="before market expiration"):
        run_backtest(pd.DataFrame([bad]), config())


def test_prediction_training_availability_cannot_be_after_prediction():
    bad = dict(row(), trained_through=pd.Timestamp("2025-01-02", tz="UTC"))
    with pytest.raises(ValueError, match="labels unavailable"):
        run_backtest(pd.DataFrame([bad]), config())


def test_simultaneous_settlements_are_one_equity_event():
    # Same-time +$4 and -$6 have net -$2. They must not create a $6 path
    # drawdown merely because one market's settlement was processed first.
    first = row("a", label=1)
    second = row("b", label=0)
    result = run_backtest(pd.DataFrame([first, second]), config())
    assert len(result.equity) == 1
    assert result.metrics["max_drawdown"] == pytest.approx(2)
