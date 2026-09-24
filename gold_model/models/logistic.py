"""Approach A: regularized logistic regression on baseline logit and few features.

Scaling uses training data only; calibration is fitted on later, disjoint markets.
Each market carries equal total sample weight despite multiple decision snapshots.
Joblib uses pickle internally: only load artifacts you created or otherwise trust.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from gold_model.models.baseline import Prediction
from gold_model.models.calibration import ProbabilityCalibrator

DEFAULT_FEATURES = ("baseline_logit", "gold_return_30s", "gold_return_1m", "volatility_5m")
FORBIDDEN_FEATURES = {"label", "result_yes", "settled_price", "settlement_value", "label_available_at", "market_end", "market_start", "market_id", "timestamp"}
DATE_COLUMNS = ("timestamp", "market_start", "market_end", "label_available_at")


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate resolved snapshots; never infer when a final label became available."""
    required = {"market_id", "label", *DATE_COLUMNS}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"Dataset must be nonempty and contain {sorted(required)}")
    result = frame.copy()
    if result["market_id"].isna().any():
        raise ValueError("Missing market ID")
    for column in DATE_COLUMNS:
        if result[column].isna().any() or any(pd.Timestamp(t).tzinfo is None for t in result[column]):
            raise ValueError(f"{column} requires explicit timezone-aware timestamps")
        result[column] = pd.to_datetime(result[column], utc=True)
    if not result["label"].isin([0, 1]).all():
        raise ValueError("Labels must be resolved binary outcomes")
    if (result["timestamp"] < result["market_start"]).any() or (result["timestamp"] >= result["market_end"]).any():
        raise ValueError("Snapshots must occur during their market window")
    if (result["label_available_at"] < result["market_end"]).any():
        raise ValueError("Labels cannot be known before market expiration")
    for column in ("label", "market_start", "market_end", "label_available_at"):
        if (result.groupby("market_id")[column].nunique() > 1).any():
            raise ValueError(f"Inconsistent {column} within a market")
    if result.duplicated(["market_id", "timestamp"]).any():
        raise ValueError("Duplicate market/timestamp snapshots")
    return result.sort_values(["timestamp", "market_id"]).reset_index(drop=True)


def market_weights(frame: pd.DataFrame) -> np.ndarray:
    """One unit of total likelihood weight per market."""
    return (1.0 / frame.groupby("market_id")["market_id"].transform("size")).to_numpy()


def _guard_later(earlier: pd.DataFrame, later: pd.DataFrame) -> None:
    if set(earlier["market_id"]) & set(later["market_id"]):
        raise ValueError("A market cannot appear in multiple periods")
    if earlier["timestamp"].max() >= later["timestamp"].min():
        raise ValueError("Periods must be strictly chronological")
    if earlier["label_available_at"].max() > later["timestamp"].min():
        raise ValueError("Earlier-period labels were unavailable at the next period's start")


def _purge_before(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    groups = frame.groupby("market_id").agg(last_label=("label_available_at", "max"), last_snapshot=("timestamp", "max"))
    ids = groups.index[(groups.last_label <= cutoff) & (groups.last_snapshot < cutoff)]
    return frame.loc[frame.market_id.isin(ids)].copy()


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    purged_market_ids: tuple[str, ...] = ()


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.6, calibration_fraction: float = 0.2) -> DatasetSplit:
    data = validate_dataset(frame)
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1 or train_fraction + calibration_fraction >= 1:
        raise ValueError("Train and calibration fractions must leave a nonempty test period")
    markets = data.groupby("market_id")["market_start"].min().sort_values(kind="stable").index
    n_train = int(len(markets) * train_fraction)
    n_cal = int(len(markets) * calibration_fraction)
    if min(n_train, n_cal, len(markets) - n_train - n_cal) < 1:
        raise ValueError("Insufficient markets for three chronological periods")
    train = data[data.market_id.isin(markets[:n_train])]
    cal = data[data.market_id.isin(markets[n_train:n_train + n_cal])]
    test = data[data.market_id.isin(markets[n_train + n_cal:])]
    cal = _purge_before(cal, test.timestamp.min())
    if cal.empty:
        raise ValueError("No calibration markets remain after label-availability purging")
    train = _purge_before(train, cal.timestamp.min())
    if train.empty:
        raise ValueError("No training markets remain after label-availability purging")
    _guard_later(train, cal)
    _guard_later(cal, test)
    retained = set(train.market_id) | set(cal.market_id) | set(test.market_id)
    return DatasetSplit(train.copy(), cal.copy(), test.copy(), tuple(sorted(set(data.market_id) - retained)))


class LogisticModel:
    def __init__(self, feature_names: Sequence[str] | None = None, calibration_method: str = "sigmoid", C: float = 1.0) -> None:
        self.feature_names = list(DEFAULT_FEATURES if feature_names is None else feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Provide unique predictor feature names")
        if FORBIDDEN_FEATURES.intersection(self.feature_names):
            raise ValueError("Outcome, identifier, and period columns cannot be model features")
        if not np.isfinite(C) or C <= 0:
            raise ValueError("C must be positive")
        self.C = C
        self.scaler = StandardScaler()
        self.estimator = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=0)
        self.calibrator = ProbabilityCalibrator(calibration_method)
        self.metadata: dict = {}
        self.version = "unfitted"

    def _features(self, frame: pd.DataFrame) -> np.ndarray:
        data = frame.copy()
        if "baseline_logit" in self.feature_names:
            if "baseline_probability" not in data:
                raise ValueError("Required feature baseline_probability is absent")
            p = pd.to_numeric(data["baseline_probability"], errors="coerce")
            if not p.between(0, 1).all():
                raise ValueError("baseline_probability must be finite and between zero and one")
            data["baseline_logit"] = logit(np.clip(p.to_numpy(dtype=float), 1e-6, 1 - 1e-6))
        missing = set(self.feature_names) - set(data.columns)
        if missing:
            raise ValueError(f"Required model features absent: {sorted(missing)}")
        values = data[self.feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            invalid = [name for name, valid in zip(self.feature_names, np.isfinite(values).all(axis=0)) if not valid]
            raise ValueError(f"Required model features unavailable: {invalid}; no zero substitution is allowed")
        return values

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame) -> "LogisticModel":
        train, calibration = validate_dataset(train), validate_dataset(calibration)
        _guard_later(train, calibration)
        if set(train.label.unique()) != {0, 1}:
            raise ValueError("Training period must contain both outcomes")
        x_train, x_cal = self._features(train), self._features(calibration)
        weights = market_weights(train)
        self.scaler.fit(x_train, sample_weight=weights)
        self.estimator.fit(self.scaler.transform(x_train), train.label.to_numpy(dtype=int), sample_weight=weights)
        raw_cal = self.estimator.predict_proba(self.scaler.transform(x_cal))[:, 1]
        self.calibrator.fit(raw_cal, calibration.label.to_numpy(dtype=int), market_weights(calibration))
        now = datetime.now(UTC)
        self.version = f"logistic-{now.strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
        from gold_model.backtesting.metrics import probability_metrics

        self.metadata = {
            "model_type": "logistic_baseline_feature", "model_version": self.version,
            "created_at": now.isoformat(), "feature_list": self.feature_names,
            "hyperparameters": {"C": self.C, "solver": "lbfgs", "max_iter": 2000},
            "calibration_method": self.calibrator.method,
            "training_start": train.timestamp.min().isoformat(), "training_end": train.timestamp.max().isoformat(),
            "training_labels_available_at": train.label_available_at.max().isoformat(),
            "calibration_start": calibration.timestamp.min().isoformat(), "calibration_end": calibration.timestamp.max().isoformat(),
            "calibration_labels_available_at": calibration.label_available_at.max().isoformat(),
            "evaluation_cutoff": calibration.label_available_at.max().isoformat(),
            "training_market_ids": sorted(train.market_id.unique().tolist()),
            "calibration_market_ids": sorted(calibration.market_id.unique().tolist()),
            "training_markets": int(train.market_id.nunique()), "calibration_markets": int(calibration.market_id.nunique()),
            "sample_weighting": "Each market has total weight one; snapshots are correlated, not independent trials",
            "validation_metrics": probability_metrics(calibration.label.to_numpy(), self.calibrator.predict(raw_cal), calibration.market_id.to_numpy()),
            "validation_metrics_note": "Calibration fit-period diagnostics; not held-out performance",
        }
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.version == "unfitted":
            raise ValueError("Model has not been fitted")
        raw = self.estimator.predict_proba(self.scaler.transform(self._features(frame)))[:, 1]
        return self.calibrator.predict(raw)

    def predict(self, features: Mapping[str, object]) -> Prediction:
        return Prediction(float(self.predict_proba(pd.DataFrame([dict(features)]))[0]))

    def save(self, path: str | Path) -> Path:
        if not self.metadata:
            raise ValueError("Cannot save an unfitted model")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        target.with_suffix(target.suffix + ".json").write_text(json.dumps(self.metadata, indent=2, allow_nan=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path, *, trusted: bool = False) -> "LogisticModel":
        if not trusted:
            raise ValueError("Joblib can execute code. Pass trusted=True only for an artifact you trust.")
        model = joblib.load(path)
        if not isinstance(model, cls) or not model.metadata:
            raise ValueError("Artifact is not a fitted gold logistic model")
        return model


def train_logistic(frame: pd.DataFrame, **kwargs) -> tuple[LogisticModel, DatasetSplit]:
    split = chronological_split(frame)
    model = LogisticModel(**kwargs).fit(split.train, split.calibration)
    model.metadata["test_start"] = split.test.timestamp.min().isoformat()
    model.metadata["test_end"] = split.test.timestamp.max().isoformat()
    model.metadata["test_markets"] = int(split.test.market_id.nunique())
    model.metadata["test_evaluation"] = "Reserved; not evaluated during training. Use evaluate/backtest explicitly."
    model.metadata["purged_market_ids"] = list(split.purged_market_ids)
    return model, split
