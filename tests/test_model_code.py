"""Unit test kode training, validasi, dan registry (tanpa jaringan)."""

from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import registry_tool
import train
import validate_model

CLASSES = ["Naik", "Turun", "Stabil"]


def make_features(n_hours=300, coins=("bitcoin", "ethereum"), seed=0):
    """Dataset fitur sintetis dengan sinyal lemah pada feat_a."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2026-01-01", periods=n_hours, freq="h", tz="UTC")
    frames = []
    for coin in coins:
        feat_a = rng.normal(size=n_hours)
        score = feat_a + rng.normal(scale=0.5, size=n_hours)
        label = np.where(score > 0.6, "Naik",
                         np.where(score < -0.6, "Turun", "Stabil"))
        frames.append(pd.DataFrame({
            "timestamp": stamps, "coin_id": coin, "price_usd": 100.0,
            "feat_a": feat_a, "feat_b": rng.normal(size=n_hours),
            "future_return_pct": score, "label": label}))
    return pd.concat(frames).reset_index(drop=True)


# ------------------------------------------------------------------ dataset
def test_load_dataset_builds_features_and_targets(tmp_path):
    path = tmp_path / "features.parquet"
    make_features().to_parquet(path, index=False)
    df, features = train.load_dataset(path, CLASSES)
    assert features == ["feat_a", "feat_b"]
    assert set(df["target"]) <= {0, 1, 2}
    assert df["timestamp"].is_monotonic_increasing


def test_load_dataset_rejects_unknown_label(tmp_path):
    data = make_features()
    data.loc[0, "label"] = "Aneh"
    path = tmp_path / "features.parquet"
    data.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        train.load_dataset(path, CLASSES)


def test_load_dataset_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        train.load_dataset(tmp_path / "tidak_ada.parquet", CLASSES)


def test_time_split_has_no_overlap_and_keeps_order(tmp_path):
    path = tmp_path / "features.parquet"
    make_features().to_parquet(path, index=False)
    df, _ = train.load_dataset(path, CLASSES)
    train_df, test_df = train.time_split(df, 0.2)
    gap = test_df["timestamp"].min() - train_df["timestamp"].max()
    assert gap >= pd.Timedelta(hours=1)
    assert len(train_df) > len(test_df) > 0


# --------------------------------------------------------- model dan metrik
def test_compute_metrics_perfect_prediction():
    y = pd.Series([0, 1, 2, 0, 1, 2])
    metrics = train.compute_metrics(y, y, CLASSES, "test")
    assert metrics["test_f1_macro"] == 1.0
    assert metrics["test_accuracy"] == 1.0
    assert {"test_f1_Naik", "test_f1_Turun", "test_f1_Stabil"} <= set(metrics)


@pytest.mark.parametrize("model, expected", [
    ("dummy", DummyClassifier),
    ("logistic_regression", Pipeline),
    ("xgboost", XGBClassifier),
])
def test_build_model_returns_expected_type(model, expected):
    args = train.parse_args(["--model", model])
    assert isinstance(train.build_model(args), expected)
    if model == "logistic_regression":
        assert isinstance(args.c, float)
        assert isinstance(train.build_model(args)[-1], LogisticRegression)


def test_config_provides_training_and_validation_sections():
    cfg = train.load_config()
    training = cfg["training"]
    args = train.apply_training_config(train.parse_args([]))
    assert args.max_depth == training["max_depth"]
    assert args.class_weight == training["class_weight"]
    assert set(cfg["validation"]) >= {
        "min_f1_gain_over_baseline", "min_class_f1",
        "max_latency_seconds", "max_train_test_gap"}


def test_apply_training_config_rejects_unknown_key(monkeypatch):
    monkeypatch.setattr(train, "load_config",
                        lambda: {"training": {"kunci_aneh": 1}})
    with pytest.raises(ValueError):
        train.apply_training_config(train.parse_args([]))


# ----------------------------------------------------------------- validasi
THRESHOLDS = {"min_f1_gain_over_baseline": 0.05, "min_class_f1": 0.25,
              "max_train_test_gap": 0.50, "max_latency_seconds": 5.0}
BASELINE = {"test_f1_macro": 0.23, "test_accuracy": 0.54}


def good_metrics(**override):
    """Metrik model yang sehat; nilai dapat ditimpa per kasus uji."""
    metrics = {"test_f1_macro": 0.41, "train_f1_macro": 0.72,
               "test_accuracy": 0.43, "test_f1_Naik": 0.35,
               "test_f1_Turun": 0.33, "test_f1_Stabil": 0.54}
    metrics.update(override)
    return metrics


def evaluate(metrics, latency=0.01):
    return validate_model.evaluate_thresholds(
        metrics, BASELINE, latency, THRESHOLDS, CLASSES)


def blocking_failures(checks):
    return [c["name"] for c in checks if c["blocking"] and not c["passed"]]


def test_validation_passes_for_good_model():
    checks, passed = evaluate(good_metrics())
    assert passed and blocking_failures(checks) == []


def test_accuracy_below_baseline_is_informational_only():
    checks, passed = evaluate(good_metrics(test_accuracy=0.43))
    accuracy = [c for c in checks if c["name"] == "test_accuracy"][0]
    assert passed
    assert not accuracy["passed"] and not accuracy["blocking"]
    assert validate_model.status_label(accuracy).startswith("INFO")


def test_validation_fails_on_overfitting_gap():
    checks, passed = evaluate(good_metrics(train_f1_macro=0.94))
    assert not passed
    assert blocking_failures(checks) == ["gap_train_test"]


def test_validation_fails_when_f1_does_not_beat_baseline():
    checks, passed = evaluate(good_metrics(test_f1_macro=0.25))
    assert not passed
    assert blocking_failures(checks) == ["test_f1_macro"]


def test_validation_fails_when_a_class_collapses():
    checks, passed = evaluate(good_metrics(test_f1_Naik=0.0))
    assert not passed
    assert blocking_failures(checks) == ["min_class_f1"]


def test_validation_fails_when_latency_too_high():
    checks, passed = evaluate(good_metrics(), latency=6.0)
    assert not passed
    assert blocking_failures(checks) == ["latency_seconds"]


def test_validation_requires_all_metrics():
    with pytest.raises(ValueError):
        evaluate({"test_f1_macro": 0.4})


# ------------------------------------------------------------------ registry
def test_parse_value_converts_numbers_only():
    assert registry_tool.parse_value("300") == 300
    assert registry_tool.parse_value("0.03") == 0.03
    assert registry_tool.parse_value("balanced") == "balanced"


def test_alias_map_groups_aliases_by_version():
    class FakeClient:
        def get_registered_model(self, name):
            return SimpleNamespace(aliases={"production": 2, "staging": 1})

    assert registry_tool.alias_map(FakeClient(), "m") == {
        "2": ["production"], "1": ["staging"]}


def test_normalize_stage():
    assert registry_tool.normalize_stage("staging") == "Staging"
    with pytest.raises(ValueError):
        registry_tool.normalize_stage("Archived")


# ---------------------------------------------- rantai training -> validasi
def test_training_then_validation_end_to_end(tmp_path, monkeypatch):
    """Smoke test: baseline + model -> MLflow -> validate_model."""
    data_path = tmp_path / "features.parquet"
    make_features().to_parquet(data_path, index=False)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    common = ["--data", str(data_path), "--tracking-uri", uri,
              "--experiment-name", "smoke", "--cv-splits", "3"]

    train.run_training(train.parse_args(
        ["--model", "dummy", "--run-name", "smoke-run-baseline"] + common))
    run_id, metrics = train.run_training(train.parse_args(
        ["--model", "xgboost", "--n-estimators", "5", "--max-depth", "2",
         "--run-name", "smoke-run"] + common))

    assert 0.0 <= metrics["test_f1_macro"] <= 1.0
    client = mlflow.MlflowClient()
    assert client.get_run(run_id).data.tags["model_uri"]

    code = validate_model.main(["--run-name", "smoke-run", "--tracking-uri",
                                uri, "--experiment-name", "smoke"])
    assert code in (0, 1)
    assert client.get_run(run_id).data.tags["validation"] in ("passed", "failed")
