"""Reproducible command workflows; held-out data never fits a saved model."""

import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from gold_model.backtesting.engine import BacktestConfig, run_backtest, threshold_sweep
from gold_model.backtesting.metrics import compare_probabilities
from gold_model.backtesting.walk_forward import walk_forward_predict
from gold_model.config import Settings
from gold_model.data.artifacts import read_dataset, write_json
from gold_model.models.baseline import BaselineModel
from gold_model.models.logistic import LogisticModel, chronological_split, train_logistic
from gold_model.services.predictor import cost_config, sizing_config
from gold_model.utils.time import utc_now

logger = logging.getLogger(__name__)


def load_model(path: Path) -> LogisticModel:
    if not path.exists():
        raise ValueError(f"Model not found: {path}. Run train logistic first, or select --model baseline.")
    # CLI paths are operator-selected, trusted local artifacts. See README: never
    # point this command at downloaded/untrusted pickle or joblib files.
    return LogisticModel.load(path, trusted=True)


def train(settings: Settings, kind: str, *, features: list[str] | None = None,
          calibration: str = "sigmoid") -> dict:
    if kind == "baseline":
        metadata = {
            "model_type": "normal_baseline", "model_version": BaselineModel.version,
            "created_at": utc_now(), "training_period": None,
            "features": ["current_gold_price", "target_price", "seconds_remaining", "volatility_5m"],
            "parameters": {"distribution": "driftless normal arithmetic diffusion", "epsilon": 1e-6},
            "note": "Analytical model; no parameters fitted to outcomes",
        }
        write_json(settings.model_path.parent / "baseline.json", metadata)
        return metadata
    if kind != "logistic":
        raise ValueError("Model must be baseline or logistic")
    frame = read_dataset(settings.dataset_path)
    model, _ = train_logistic(frame, feature_names=features, calibration_method=calibration)
    model.metadata["dataset_provenance"] = frame.attrs.get("manifest", {})
    model.save(settings.model_path)
    logger.info("training_run", extra={"context": {"version": model.version, "path": str(settings.model_path)}})
    return model.metadata


def heldout_predictions(settings: Settings, kind: str = "logistic") -> pd.DataFrame:
    frame = read_dataset(settings.dataset_path)
    if kind == "logistic":
        model = load_model(settings.model_path)
        used_ids = set(model.metadata["training_market_ids"]) | set(model.metadata["calibration_market_ids"])
        cutoff = pd.Timestamp(model.metadata["evaluation_cutoff"])
        # Market starts must also be out of period; never score a partially
        # overlapping market just because one late snapshot passed the cutoff.
        groups = frame.groupby("market_id").agg(start=("market_start", "min"))
        eligible_ids = set(groups.index[groups.start >= cutoff]) - used_ids
        frame = frame[frame.market_id.isin(eligible_ids)].copy()
        if frame.empty:
            raise ValueError("No held-out markets after the model's calibration label cutoff")
        frame["probability_yes"] = model.predict_proba(frame)
        frame.attrs["model_version"] = model.version
    else:
        frame = chronological_split(frame).test.copy()
        if kind == "baseline":
            frame["probability_yes"] = BaselineModel().predict_proba(frame)
        elif kind == "market":
            frame = frame[frame.kalshi_mid_probability.notna()].copy()
            frame["probability_yes"] = frame.kalshi_mid_probability
        else:
            raise ValueError("Model must be baseline, logistic, or market")
        if frame.empty:
            raise ValueError("No eligible held-out predictions")
        frame.attrs["model_version"] = kind
    return frame


def evaluate(settings: Settings, kind: str = "logistic") -> dict:
    frame = heldout_predictions(settings, kind)
    report = {
        "evaluation": "chronological held-out markets",
        "model_version": frame.attrs.get("model_version"),
        "first_prediction": frame.timestamp.min(), "last_prediction": frame.timestamp.max(),
        "probability_comparison": compare_probabilities(frame),
        "provenance": frame.attrs.get("manifest"),
    }
    write_json(settings.report_dir / "evaluation.json", report)
    return report


def backtest(settings: Settings, kind: str = "logistic", *, walk_forward: bool = False,
             idealized: bool = False, min_train_markets: int = 20,
             calibration_markets: int = 5, test_markets: int = 5,
             features: list[str] | None = None) -> dict:
    folds = None
    if walk_forward:
        if kind != "logistic":
            raise ValueError("Walk-forward fitting requires --model logistic")
        dataset = read_dataset(settings.dataset_path)
        result = walk_forward_predict(dataset, min_train_markets=min_train_markets,
                                      calibration_markets=calibration_markets,
                                      test_markets=test_markets, features=features)
        frame, folds = result.predictions, result.folds
        frame.attrs["manifest"] = dataset.attrs.get("manifest")
    else:
        frame = heldout_predictions(settings, kind)
    config = BacktestConfig(
        initial_bankroll=settings.bankroll, min_edge=settings.min_edge,
        max_book_age_seconds=settings.book_max_age_seconds,
        mode="idealized" if idealized else "executable",
        costs=cost_config(settings), sizing=sizing_config(settings),
    )
    result = run_backtest(frame, config)
    sweep = threshold_sweep(frame, config=config)
    comparison = {}
    common = frame.dropna(subset=["baseline_probability", "probability_yes", "kalshi_mid_probability"])
    if not common.empty:
        for name, column in (("baseline", "baseline_probability"), ("logistic" if kind == "logistic" else "selected", "probability_yes"), ("market", "kalshi_mid_probability")):
            candidate = common.copy()
            candidate["probability_yes"] = common[column]
            comparison[name] = run_backtest(candidate, replace(config)).metrics
    report = {
        "evaluation": "walk-forward out-of-sample" if walk_forward else "chronological held-out markets",
        "model_version": frame.attrs.get("model_version", "fold-specific logistic models"),
        "metrics": result.metrics, "thresholds": sweep.to_dict("records"),
        "strategy_comparison_common_support": comparison,
        "folds": folds, "provenance": frame.attrs.get("manifest"),
        "threshold_note": "This sweep is descriptive. Do not choose a threshold from final-test P&L; preselect it on validation data for a later untouched period.",
        "limits": "Historical displayed asks are execution estimates, not guaranteed fills. Drawdown uses settled equity. No claim of future profitability.",
    }
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    write_json(settings.report_dir / "backtest.json", report)
    result.trades.to_csv(settings.report_dir / "trades.csv", index=False)
    result.decisions.to_csv(settings.report_dir / "decisions.csv", index=False)
    result.equity.to_csv(settings.report_dir / "equity.csv", index=False)
    sweep.to_csv(settings.report_dir / "thresholds.csv", index=False)
    frame.to_csv(settings.report_dir / "oos_predictions.csv", index=False)
    logger.info("backtest_result", extra={"context": {"evaluation": report["evaluation"], "report": str(settings.report_dir / "backtest.json")}})
    return report
