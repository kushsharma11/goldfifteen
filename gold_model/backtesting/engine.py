"""Event-ordered, one-entry-per-market research simulator.

Only visible top-of-book size is executable. Entry cash remains reserved until
label_available_at. Missing books are never replaced by midpoint fills unless an
entire run explicitly uses idealized mode.
"""

from dataclasses import dataclass, field, replace
from collections import deque
import heapq
from math import isfinite

import numpy as np
import pandas as pd

from gold_model.backtesting.metrics import compare_probabilities, trading_metrics
from gold_model.models.logistic import validate_dataset
from gold_model.trading.edge import CostConfig, calculate_signal
from gold_model.trading.sizing import SizingConfig, size_position


@dataclass(frozen=True)
class BacktestConfig:
    initial_bankroll: float = 1000.0
    min_edge: float = 0.05
    mode: str = "executable"
    max_book_age_seconds: float = 30.0
    costs: CostConfig = field(default_factory=CostConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)

    def __post_init__(self) -> None:
        if not isfinite(self.initial_bankroll) or self.initial_bankroll <= 0:
            raise ValueError("Initial bankroll must be positive")
        if self.mode not in {"executable", "idealized"}:
            raise ValueError("Backtest mode must be executable or idealized")
        if not isfinite(self.min_edge) or not 0 <= self.min_edge <= 1:
            raise ValueError("Minimum edge must be between zero and one")
        if not isfinite(self.max_book_age_seconds) or self.max_book_age_seconds < 0:
            raise ValueError("Maximum book age must be finite and nonnegative")


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    decisions: pd.DataFrame
    metrics: dict
    equity: pd.DataFrame


def _number(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if isfinite(value) else None


def run_backtest(frame: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    config = config or BacktestConfig()
    data = validate_dataset(frame)
    if "probability_yes" not in data:
        raise ValueError("Backtest requires out-of-sample probability_yes predictions")
    probabilities = pd.to_numeric(data.probability_yes, errors="coerce")
    if not probabilities.between(0, 1).all():
        raise ValueError("Every backtest row requires a finite model probability")
    if "trained_through" in data:
        if data.trained_through.isna().any() or any(pd.Timestamp(value).tzinfo is None for value in data.trained_through):
            raise ValueError("trained_through must contain explicit timezone-aware availability cutoffs")
        if (pd.to_datetime(data.trained_through, utc=True) > data.timestamp).any():
            raise ValueError("Predictions were trained on labels unavailable at prediction time")
    cash = config.initial_bankroll
    settled_equity = config.initial_bankroll
    pending: list[tuple[pd.Timestamp, int]] = []
    trades, decisions, equity = [], [], []
    traded_markets = set()
    previous_volatility: deque[float] = deque(maxlen=1000)

    def settle_through(now: pd.Timestamp | None) -> None:
        nonlocal cash, settled_equity
        while pending and (now is None or pending[0][0] <= now):
            when = pending[0][0]
            while pending and pending[0][0] == when:
                _, index = heapq.heappop(pending)
                trade = trades[index]
                cash += trade["payout"]
                settled_equity += trade["pnl"]
            equity.append({"timestamp": when.isoformat(), "settlement_equity": settled_equity, "available_cash": cash})

    for row in data.to_dict("records"):
        now, market_id = row["timestamp"], row["market_id"]
        settle_through(now)
        remaining = (row["market_end"] - now).total_seconds()
        vol = _number(row, "volatility_5m")
        regime = "unknown"
        if vol is not None and len(previous_volatility) >= 10:
            low, high = np.quantile(previous_volatility, [.33, .67])
            regime = "low" if vol < low else "high" if vol > high else "normal"
        if vol is not None:
            previous_volatility.append(vol)
        reason = None
        actionable = remaining > 0 and market_id not in traded_markets
        if market_id in traded_markets:
            reason = "Already traded this market"
        elif remaining <= 0:
            reason = "Market expired"
        if config.mode == "idealized":
            yes_ask = _number(row, "kalshi_mid_probability")
            no_ask = None if yes_ask is None else 1 - yes_ask
            yes_depth = no_depth = None
        else:
            age = _number(row, "book_age_seconds")
            if age is None or age < 0 or age > config.max_book_age_seconds:
                actionable, reason = False, "Missing or stale historical order book"
            yes_ask, no_ask = _number(row, "kalshi_yes_ask"), _number(row, "kalshi_no_ask")
            yes_depth, no_depth = _number(row, "yes_ask_size"), _number(row, "no_ask_size")
            if yes_depth is None or yes_depth < 1:
                yes_ask = None
            if no_depth is None or no_depth < 1:
                no_ask = None
        signal = calculate_signal(row["probability_yes"], yes_ask, no_ask, config.min_edge, config.costs, actionable, reason)
        position = size_position(signal, cash, config.costs, config.sizing, yes_depth if signal.action == "YES" else no_depth)
        action = signal.action if position.contracts else "PASS"
        decisions.append({"timestamp": now.isoformat(), "market_id": market_id, "action": action,
                          "reason": signal.reason if action == "PASS" and signal.action == "PASS" else "Exposure, cash, or depth limit" if action == "PASS" else signal.reason,
                          "yes_edge": signal.yes_edge, "no_edge": signal.no_edge, "available_cash_before": cash})
        if action == "PASS":
            continue
        cash -= position.risk_dollars
        won = bool(row["label"] == (1 if action == "YES" else 0))
        payout = float(position.contracts if won else 0)
        trade = {"timestamp": now.isoformat(), "market_id": market_id, "side": action,
                 "settled_at": row["label_available_at"].isoformat(), "label": int(row["label"]), "won": won,
                 "probability_yes": row["probability_yes"], "estimated_edge": signal.selected_edge,
                 "contracts": position.contracts, "execution_price": position.execution_price,
                 "risk_dollars": position.risk_dollars, "fee_dollars": position.fee_dollars,
                 "payout": payout, "pnl": payout - position.risk_dollars,
                 "seconds_remaining": remaining, "utc_hour": now.hour, "volatility_regime": regime}
        trades.append(trade)
        traded_markets.add(market_id)
        heapq.heappush(pending, (row["label_available_at"], len(trades) - 1))
    settle_through(None)
    trade_frame = pd.DataFrame(trades, columns=[
        "timestamp", "market_id", "side", "settled_at", "label", "won", "probability_yes",
        "estimated_edge", "contracts", "execution_price", "risk_dollars", "fee_dollars",
        "payout", "pnl", "seconds_remaining", "utc_hour", "volatility_regime",
    ])
    decisions_frame = pd.DataFrame(decisions)
    metrics = trading_metrics(trade_frame, decisions_frame, equity, config.initial_bankroll)
    metrics.update({"mode": config.mode, "fees_included": config.costs.include_fees,
                    "fee_per_contract": config.costs.fee_per_contract, "quadratic_fee_rate": config.costs.quadratic_fee_rate,
                    "fee_rounding": "Up to nearest cent per order", "slippage_cents": config.costs.slippage_cents,
                    "min_edge": config.min_edge, "probability_comparison": compare_probabilities(data, frame.attrs.get("selected_model_name", "logistic")),
                    "execution_assumption": "Top-of-book ask and displayed size; no queue position or latency guarantee" if config.mode == "executable" else "IDEALIZED midpoint fills without historical depth; not executable profitability",
                    "volatility_regime_basis": "Prior 1000 snapshots' volatility tertiles; first ten snapshots unknown",
                    "evaluation_note": "Caller must supply held-out or walk-forward predictions; thresholds are descriptive sensitivity, not optimized choices"})
    return BacktestResult(trade_frame, decisions_frame, metrics, pd.DataFrame(equity, columns=["timestamp", "settlement_equity", "available_cash"]))


def threshold_sweep(frame: pd.DataFrame, thresholds=(.01, .02, .03, .05, .07, .10, .15), config: BacktestConfig | None = None) -> pd.DataFrame:
    config = config or BacktestConfig()
    rows = []
    fields = ("trades", "win_rate", "average_estimated_edge", "average_realized_return", "average_profit_per_trade", "cumulative_pnl", "max_drawdown", "roi")
    for threshold in thresholds:
        result = run_backtest(frame, replace(config, min_edge=float(threshold)))
        rows.append({"min_edge": float(threshold), **{key: result.metrics[key] for key in fields}})
    return pd.DataFrame(rows)
