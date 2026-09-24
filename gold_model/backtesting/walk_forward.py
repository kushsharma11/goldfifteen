"""Expanding chronological train -> calibrate -> test, preserving market groups."""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from gold_model.models.logistic import LogisticModel, _guard_later, _purge_before, validate_dataset


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    folds: list[dict]


def walk_forward_predict(
    frame: pd.DataFrame,
    min_train_markets: int = 20,
    calibration_markets: int = 5,
    test_markets: int = 5,
    features: Sequence[str] | None = None,
    calibration_method: str = "sigmoid",
    C: float = 1.0,
) -> WalkForwardResult:
    data = validate_dataset(frame)
    if min(min_train_markets, calibration_markets, test_markets) < 1:
        raise ValueError("Fold sizes must be positive")
    market_ids = data.groupby("market_id").market_start.min().sort_values(kind="stable").index
    first_test = min_train_markets + calibration_markets
    if len(market_ids) <= first_test:
        raise ValueError("Not enough markets for walk-forward train, calibration, and test")
    predictions, folds = [], []
    for start in range(first_test, len(market_ids), test_markets):
        train_ids = market_ids[: start - calibration_markets]
        cal_ids = market_ids[start - calibration_markets : start]
        test_ids = market_ids[start : start + test_markets]
        train = data[data.market_id.isin(train_ids)]
        cal = data[data.market_id.isin(cal_ids)]
        test = data[data.market_id.isin(test_ids)].copy()
        cal = _purge_before(cal, test.timestamp.min())
        if cal.empty:
            raise ValueError(
                f"Fold {len(folds)} has no calibration markets after availability purging"
            )
        train = _purge_before(train, cal.timestamp.min())
        if train.empty:
            raise ValueError(
                f"Fold {len(folds)} has no training markets after availability purging"
            )
        _guard_later(cal, test)
        model = LogisticModel(features, calibration_method, C).fit(train, cal)
        test["probability_yes"] = model.predict_proba(test)
        test["model_version"] = model.version
        test["fold"] = len(folds)
        test["trained_through"] = model.metadata["evaluation_cutoff"]
        predictions.append(test)
        folds.append(
            {
                "fold": len(folds),
                "training_markets": int(train.market_id.nunique()),
                "calibration_markets": int(cal.market_id.nunique()),
                "test_markets": int(test.market_id.nunique()),
                "test_market_ids": list(test_ids),
                "test_start": test.timestamp.min().isoformat(),
                "test_end": test.timestamp.max().isoformat(),
                "model_metadata": model.metadata,
            }
        )
    return WalkForwardResult(pd.concat(predictions, ignore_index=True), folds)
