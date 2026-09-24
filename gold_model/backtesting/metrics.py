"""Calibration diagnostics and settlement-equity trading statistics."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def probability_metrics(labels, probabilities, market_ids=None, bins: int = 10) -> dict:
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Probability metrics require nonempty aligned arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid binary labels or probabilities")
    if bins < 1:
        raise ValueError("At least one reliability bin is required")
    ids = np.arange(len(y)).astype(str) if market_ids is None else np.asarray(market_ids)
    if len(ids) != len(y):
        raise ValueError("Market IDs must align with labels")
    counts = pd.Series(ids).value_counts()
    weights = np.array([1.0 / counts[item] for item in ids])
    reliability = []
    bucket = np.minimum((p * bins).astype(int), bins - 1)
    for i in range(bins):
        mask = bucket == i
        if mask.any():
            reliability.append({
                "lower": i / bins, "upper": (i + 1) / bins,
                "mean_probability": float(np.average(p[mask], weights=weights[mask])),
                "observed_yes_rate": float(np.average(y[mask], weights=weights[mask])),
                "snapshots": int(mask.sum()), "markets": int(len(set(ids[mask]))),
            })
    return {
        "brier_score": float(brier_score_loss(y, p, sample_weight=weights)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1], sample_weight=weights)),
        "accuracy": float(accuracy_score(y, p >= 0.5, sample_weight=weights)),
        "snapshots": len(y), "markets": len(counts), "reliability": reliability,
        "weighting": "Equal aggregate weight per market; correlated snapshots are not independent observations",
    }


def compare_probabilities(frame: pd.DataFrame, model_name: str = "logistic") -> dict:
    columns = {"baseline": "baseline_probability", model_name: "probability_yes", "market": "kalshi_mid_probability"}
    results = {}
    available = []
    for name, column in columns.items():
        if column not in frame:
            results[name] = {"unavailable": f"Missing {column}"}
            continue
        mask = pd.to_numeric(frame[column], errors="coerce").between(0, 1) & frame.label.isin([0, 1])
        if not mask.any():
            results[name] = {"unavailable": "No finite aligned probabilities"}
            continue
        available.append((name, column))
        rows = frame.loc[mask]
        results[name] = probability_metrics(rows.label, rows[column], rows.market_id)
    common = pd.Series(True, index=frame.index)
    for _, column in available:
        common &= pd.to_numeric(frame[column], errors="coerce").between(0, 1)
    if available and common.any():
        rows = frame.loc[common]
        results["common_support"] = {name: probability_metrics(rows.label, rows[column], rows.market_id) for name, column in available}
        shared = results["common_support"]
        if "market" in shared:
            differences = {}
            for name, values in shared.items():
                if name == "market":
                    continue
                brier_delta = values["brier_score"] - shared["market"]["brier_score"]
                loss_delta = values["log_loss"] - shared["market"]["log_loss"]
                assessment = "Mixed or tied probability results on this sample"
                if brier_delta < 0 and loss_delta < 0:
                    assessment = "Lower Brier score and log loss than market on this sample"
                elif brier_delta > 0 and loss_delta > 0:
                    assessment = "Higher Brier score and log loss than market on this sample"
                differences[name] = {"brier_difference_vs_market": brier_delta, "log_loss_difference_vs_market": loss_delta, "assessment": assessment}
            results["relative_to_market"] = differences
    results["interpretation"] = "Compare models on common_support. No outperformance claim or threshold optimization is inferred. Market midpoint is a probability benchmark, not a fill price."
    return results


def _summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0, "win_rate": None, "average_estimated_edge": None, "average_realized_return": None, "average_profit_per_trade": None, "cumulative_pnl": 0.0, "roi": None}
    spent = float(trades.risk_dollars.sum())
    return {
        "trades": len(trades), "win_rate": float(trades.won.mean()),
        "average_estimated_edge": float(trades.estimated_edge.mean()),
        "average_realized_return": float((trades.pnl / trades.risk_dollars).mean()),
        "average_profit_per_trade": float(trades.pnl.mean()),
        "cumulative_pnl": float(trades.pnl.sum()), "roi": float(trades.pnl.sum() / spent) if spent else None,
    }


def trading_metrics(trades: pd.DataFrame, decisions: pd.DataFrame, equity: list[dict], initial_bankroll: float) -> dict:
    result = _summary(trades)
    values = np.array([initial_bankroll] + [row["settlement_equity"] for row in equity], dtype=float)
    peaks = np.maximum.accumulate(values)
    result.update({
        "yes_trades": int((trades.side == "YES").sum()) if not trades.empty else 0,
        "no_trades": int((trades.side == "NO").sum()) if not trades.empty else 0,
        "pass_count": int((decisions.action == "PASS").sum()) if not decisions.empty else 0,
        "pass_markets": int(decisions.market_id.nunique() - trades.market_id.nunique()) if not trades.empty else int(decisions.market_id.nunique()),
        "max_drawdown": float((peaks - values).max()),
        "max_drawdown_fraction": float(np.max(np.divide(peaks - values, peaks, out=np.zeros_like(peaks), where=peaks > 0))),
        "drawdown_basis": "Settlement equity: initial cash plus P&L when outcomes become available; positions carried at cost, no intramarket mark-to-market",
        "bankroll_return": float(result["cumulative_pnl"] / initial_bankroll),
        "final_bankroll": float(values[-1]), "sharpe_like": None,
        "sharpe_note": "Unannualized mean trade return / sample standard deviation, only when >=30 trades; ignores serial correlation and is descriptive",
        "breakdowns": {},
    })
    if trades.empty:
        return result
    returns = (trades.pnl / trades.risk_dollars).to_numpy()
    if len(returns) >= 30 and np.std(returns, ddof=1) > 0:
        result["sharpe_like"] = float(np.mean(returns) / np.std(returns, ddof=1))
    data = trades.copy()
    data["edge_bucket"] = pd.cut(data.estimated_edge, [-np.inf, .02, .05, .10, .15, np.inf], labels=["0-2%", "2-5%", "5-10%", "10-15%", "15%+"], right=False)
    data["seconds_remaining_bucket"] = pd.cut(data.seconds_remaining, [0, 60, 180, 300, 600, np.inf], labels=["0-60", "61-180", "181-300", "301-600", "601+"], include_lowest=True)
    for column in ("edge_bucket", "seconds_remaining_bucket", "utc_hour", "volatility_regime", "side"):
        result["breakdowns"][column] = {str(key): _summary(group) for key, group in data.groupby(column, observed=True, dropna=False)}
    return result
