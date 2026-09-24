"""Probability calibration on a strictly later, disjoint market period."""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    method: str = "sigmoid"
    estimator: object | None = field(default=None, init=False)
    fitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.method not in {"sigmoid", "isotonic", "none"}:
            raise ValueError("Calibration must be sigmoid, isotonic, or none")

    def fit(
        self, probabilities: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "ProbabilityCalibrator":
        p = np.asarray(probabilities, dtype=float)
        y = np.asarray(labels, dtype=int)
        if len(p) != len(y) or not len(y) or not np.isfinite(p).all():
            raise ValueError("Calibration requires finite aligned probabilities and labels")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Calibration period must contain both resolved outcomes")
        if self.method == "sigmoid":
            self.estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            self.estimator.fit(
                logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1), y, sample_weight=sample_weight
            )
        elif self.method == "isotonic":
            self.estimator = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            self.estimator.fit(p, y, sample_weight=sample_weight)
        self.fitted = True
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Calibrator is not fitted")
        p = np.asarray(probabilities, dtype=float)
        if self.method == "sigmoid":
            p = self.estimator.predict_proba(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            p = self.estimator.predict(p)
        return np.clip(p, 1e-6, 1 - 1e-6)
