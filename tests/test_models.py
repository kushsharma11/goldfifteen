import json

import numpy as np
import pandas as pd
import pytest

from gold_model.backtesting.metrics import compare_probabilities, probability_metrics
from gold_model.backtesting.walk_forward import walk_forward_predict
from gold_model.models.baseline import BaselineModel, Prediction, baseline_probability
from gold_model.models.logistic import LogisticModel, chronological_split, train_logistic


def dataset(n=40, snapshots=2):
    start = pd.Timestamp("2025-01-01", tz="UTC")
    rows = []
    for i in range(n):
        opening = start + pd.Timedelta(minutes=15 * i)
        label = i % 2
        for j in range(snapshots):
            rows.append({
                "market_id": f"market-{i:03}", "timestamp": opening + pd.Timedelta(seconds=60 + j * 60),
                "market_start": opening, "market_end": opening + pd.Timedelta(minutes=15),
                "label_available_at": opening + pd.Timedelta(minutes=15, seconds=20), "label": label,
                "baseline_probability": .65 if label else .35,
                "gold_return_30s": .001 if label else -.001, "gold_return_1m": .002 if label else -.002,
                "volatility_5m": .1 + i % 3 * .01, "kalshi_mid_probability": .5,
            })
    return pd.DataFrame(rows)


def test_normal_baseline_symmetry_complement_expiration_and_zero_volatility():
    p = baseline_probability(4301, 4300, 100, .1)
    assert p == pytest.approx(.841344746)
    assert baseline_probability(4299, 4300, 100, .1) == pytest.approx(1 - p)
    assert baseline_probability(4300, 4300, 100, 0) == .5
    assert baseline_probability(4300, 4300, 0, .1) == 1
    assert baseline_probability(4299, 4300, 0, .1) == 0
    prediction = Prediction(p)
    assert prediction.probability_yes + prediction.probability_no == 1
    assert BaselineModel().predict({"target_z_score": 1}).probability_yes == pytest.approx(p)
    with pytest.raises(ValueError, match="finite"):
        baseline_probability(float("nan"), 4300, 100, .1)


def test_chronological_split_keeps_groups_and_purges_unavailable_labels():
    data = dataset(30)
    data.loc[data.market_id == "market-017", "label_available_at"] += pd.Timedelta(hours=3)
    split = chronological_split(data)
    assert "market-017" in split.purged_market_ids
    assert not set(split.train.market_id) & set(split.calibration.market_id)
    assert not set(split.train.market_id) & set(split.test.market_id)
    assert not set(split.calibration.market_id) & set(split.test.market_id)
    assert split.train.label_available_at.max() <= split.calibration.timestamp.min()
    assert split.calibration.label_available_at.max() <= split.test.timestamp.min()
    for part in (split.train, split.calibration, split.test):
        assert (part.groupby("market_id").size() == 2).all()


def test_train_calibrate_serialize_and_missing_features(tmp_path):
    model, split = train_logistic(dataset())
    predicted = model.predict_proba(split.test)
    assert ((predicted > 0) & (predicted < 1)).all()
    path = model.save(tmp_path / "model.joblib")
    metadata = json.loads(path.with_suffix(".joblib.json").read_text())
    assert metadata["feature_list"][0] == "baseline_logit"
    assert "test_metrics" not in metadata
    assert "not evaluated during training" in metadata["test_evaluation"]
    assert metadata["calibration_labels_available_at"] <= metadata["test_start"]
    with pytest.raises(ValueError, match="execute code"):
        LogisticModel.load(path)
    restored = LogisticModel.load(path, trusted=True)
    np.testing.assert_allclose(predicted, restored.predict_proba(split.test))
    missing = split.test.copy()
    missing.loc[missing.index[0], "gold_return_30s"] = np.nan
    with pytest.raises(ValueError, match="no zero substitution"):
        model.predict_proba(missing)


def test_fit_rejects_shared_markets_future_labels_and_outcome_features():
    split = chronological_split(dataset())
    with pytest.raises(ValueError, match="multiple periods"):
        LogisticModel().fit(split.train, split.train)
    train = split.train.copy()
    train["label_available_at"] += pd.Timedelta(days=2)
    with pytest.raises(ValueError, match="unavailable"):
        LogisticModel().fit(train, split.calibration)
    with pytest.raises(ValueError, match="Outcome"):
        LogisticModel(["label"])


def test_training_and_calibration_are_not_affected_by_test_labels():
    original = dataset()
    model, split = train_logistic(original)
    changed = original.copy()
    changed.loc[changed.market_id.isin(split.test.market_id), "label"] = 1 - changed.loc[changed.market_id.isin(split.test.market_id), "label"]
    other, _ = train_logistic(changed)
    np.testing.assert_allclose(model.predict_proba(split.test), other.predict_proba(split.test))


def test_training_does_not_score_reserved_test_rows():
    original = dataset()
    split = chronological_split(original)
    original.loc[original.market_id.isin(split.test.market_id), "gold_return_30s"] = np.nan
    model, _ = train_logistic(original)
    assert "test_metrics" not in model.metadata


def test_equal_market_weighting_and_probability_comparison():
    metrics = probability_metrics([1, 1, 0], [.8, .8, .8], ["a", "a", "b"])
    assert metrics["brier_score"] == pytest.approx((.04 + .64) / 2)
    frame = dataset(4)
    frame["probability_yes"] = frame.baseline_probability
    frame.loc[0, "kalshi_mid_probability"] = np.nan
    comparison = compare_probabilities(frame)
    assert comparison["common_support"]["logistic"]["snapshots"] == len(frame) - 1
    assert comparison["common_support"]["market"]["snapshots"] == len(frame) - 1
    assert comparison["relative_to_market"]["logistic"]["brier_difference_vs_market"] < 0
    baseline = compare_probabilities(frame, "baseline")
    assert "logistic" not in baseline


def test_walk_forward_only_predicts_later_disjoint_markets():
    result = walk_forward_predict(dataset(30), min_train_markets=10, calibration_markets=4, test_markets=4)
    assert len(result.folds) == 4
    assert result.predictions.market_id.nunique() == 16
    assert not result.predictions.duplicated(["market_id", "timestamp"]).any()
    for fold in result.folds:
        metadata = fold["model_metadata"]
        assert pd.Timestamp(metadata["evaluation_cutoff"]) <= pd.Timestamp(fold["test_start"])
        assert not set(fold["test_market_ids"]) & set(metadata["training_market_ids"])
        assert not set(fold["test_market_ids"]) & set(metadata["calibration_market_ids"])
