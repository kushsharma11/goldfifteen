"""Drift-free normal baseline. Prices and remaining standard deviation are USD."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
import pandas as pd
from scipy.special import ndtr


@dataclass(frozen=True)
class Prediction:
    probability_yes: float

    def __post_init__(self) -> None:
        if not isfinite(self.probability_yes) or not 0 <= self.probability_yes <= 1:
            raise ValueError("probability_yes must be finite and between 0 and 1")

    @property
    def probability_no(self) -> float:
        return 1.0 - self.probability_yes


def baseline_probability(
    current_price: float,
    target_price: float,
    seconds_remaining: float,
    volatility_per_sqrt_second: float,
    *,
    minimum_volatility: float = 1e-6,
    epsilon: float = 1e-6,
) -> float:
    """P(final >= target), assuming independent normal USD price increments.

    At expiration this describes the supplied final price, not an actionable
    prediction. Zero estimated volatility is floored rather than divided by zero.
    """
    values = (current_price, target_price, seconds_remaining, volatility_per_sqrt_second)
    if not all(isfinite(v) for v in values):
        raise ValueError("Baseline inputs must be finite")
    if current_price <= 0 or target_price <= 0 or volatility_per_sqrt_second < 0:
        raise ValueError("Prices must be positive and volatility nonnegative")
    if not 0 < epsilon < 0.5 or not isfinite(minimum_volatility) or minimum_volatility <= 0:
        raise ValueError("Invalid numerical safeguards")
    if seconds_remaining <= 0:
        return float(current_price >= target_price)
    move = max(volatility_per_sqrt_second, minimum_volatility) * sqrt(seconds_remaining)
    return float(np.clip(ndtr((current_price - target_price) / move), epsilon, 1 - epsilon))


class BaselineModel:
    version = "normal-baseline-v1"

    def predict(self, features: Mapping[str, object]) -> Prediction:
        if "baseline_probability" in features:
            return Prediction(float(features["baseline_probability"]))
        if "target_z_score" in features:
            z = float(features["target_z_score"])
            if not isfinite(z):
                raise ValueError("target_z_score is missing or not finite")
            return Prediction(float(np.clip(ndtr(z), 1e-6, 1 - 1e-6)))
        return Prediction(
            baseline_probability(
                float(features["current_gold_price"]),
                float(features["target_price"]),
                float(features["seconds_remaining"]),
                float(features["volatility_5m"]),
            )
        )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray([self.predict(row).probability_yes for row in frame.to_dict("records")])
