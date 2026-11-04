"""
Training model klasifikasi tren harga (Naik / Turun / Stabil) dengan MLflow.

Setiap eksekusi = satu run MLflow yang mencatat:
    * parameter  : hyperparameter, seed, ukuran data, versi data (md5 + tag)
    * metrik     : walk-forward CV (data latih) dan evaluasi hold-out (data uji)
    * artefak    : model (log_model), confusion matrix, classification report
    * tag        : git commit, versi data, pemetaan label

Evaluasi mengikuti urutan waktu (bukan acak) agar tidak terjadi kebocoran
data masa depan: 80% data tertua untuk latih (dengan walk-forward CV),
20% data terbaru untuk uji akhir.
"""

import argparse
import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "dvc" / "features_train.parquet"
DEFAULT_EXPERIMENT = "PredictaCoin-trend-classification"
NON_FEATURE_COLS = {"timestamp", "coin_id", "price_usd",
                    "future_return_pct", "label"}
MODELS = ("dummy", "logistic_regression", "xgboost")

logger = logging.getLogger("train")


def load_config():
    """Baca config/config.yaml dan kembalikan seluruh isinya sebagai dict."""
    with open(PROJECT_ROOT / "config" / "config.yaml",
              encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_tracking_uri():
    """URI tracking MLflow: env MLFLOW_TRACKING_URI, atau SQLite di root proyek.

    SQLite dipilih karena Model Registry membutuhkan database backend.
    """
    return os.getenv("MLFLOW_TRACKING_URI",
                     f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")


def file_md5(path):
    """Hitung md5 file (sama dengan hash yang dicatat DVC)."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info():
    """Kembalikan (commit singkat, apakah ada perubahan belum di-commit)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except Exception:
        return "unknown", False


def load_dataset(path, classes):
    """Muat fitur dan label; kembalikan (DataFrame, daftar nama fitur).

    Raises:
        FileNotFoundError: dataset tidak ada.
        ValueError: kolom wajib hilang atau ada label di luar `classes`.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}. Buat snapshot fitur atau "
            "jalankan 'dvc checkout' (lihat README).")
    df = pd.read_parquet(path)
    missing = {"timestamp", "coin_id", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib hilang: {sorted(missing)}")

    features = [c for c in df.columns if c not in NON_FEATURE_COLS]
    df = df.dropna(subset=features + ["label"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["timestamp", "coin_id"]).reset_index(drop=True)
    df[features] = df[features].astype(float)

    label_to_id = {name: i for i, name in enumerate(classes)}
    unknown = set(df["label"].unique()) - set(label_to_id)
    if unknown:
        raise ValueError(f"Label tidak dikenal: {sorted(unknown)}")
    df["target"] = df["label"].map(label_to_id)
    return df, features


def time_split(df, test_size):
    """Pisah kronologis: data terbaru untuk uji, 1 jam pemisah (purge)."""
    stamps = df["timestamp"].drop_duplicates().sort_values()
    cutoff = stamps.iloc[int(len(stamps) * (1 - test_size))]
    train = df[df["timestamp"] < cutoff - pd.Timedelta(hours=1)]
    test = df[df["timestamp"] >= cutoff]
    return train, test


def build_model(args):
    """Bangun model sesuai argumen CLI."""
    if args.model == "dummy":
        return DummyClassifier(strategy="most_frequent")
    class_weight = "balanced" if args.class_weight == "balanced" else None
    if args.model == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=args.c, max_iter=1000,
                               class_weight=class_weight,
                               random_state=args.seed))
    return XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", eval_metric="mlogloss",
        random_state=args.seed, n_jobs=2)


def model_params(args):
    """Hyperparameter yang relevan untuk model terpilih (dicatat ke MLflow)."""
    params = {"model": args.model, "class_weight": args.class_weight}
    if args.model == "logistic_regression":
        params["C"] = args.c
    if args.model == "xgboost":
        params.update(n_estimators=args.n_estimators,
                      max_depth=args.max_depth,
                      learning_rate=args.learning_rate)
    return params


def fit_model(model, args, x, y):
    """Latih model; XGBoost memakai sample_weight untuk class balancing."""
    if args.model == "xgboost" and args.class_weight == "balanced":
        model.fit(x, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(x, y)
    return model


def walk_forward_cv(model, args, x, y, n_classes):
    """Walk-forward CV pada data latih; kembalikan (mean, std) F1-macro."""
    scores = []
    for train_idx, val_idx in TimeSeriesSplit(
            n_splits=args.cv_splits).split(x):
        if len(np.unique(y.iloc[train_idx])) < n_classes:
            continue  # lipatan awal belum memuat semua kelas
        fold = fit_model(clone(model), args, x.iloc[train_idx],
                         y.iloc[train_idx])
        pred = fold.predict(x.iloc[val_idx])
        scores.append(f1_score(y.iloc[val_idx], pred, average="macro",
                               zero_division=0))
    if not scores:
        raise ValueError("Data terlalu sedikit untuk walk-forward CV.")
    return float(np.mean(scores)), float(np.std(scores))


def compute_metrics(y_true, y_pred, classes, prefix):
    """Hitung metrik klasifikasi; nama metrik diawali `prefix`."""
    metrics = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_f1_macro": f1_score(y_true, y_pred, average="macro",
                                       zero_division=0),
        f"{prefix}_f1_weighted": f1_score(y_true, y_pred,
                                          average="weighted",
                                          zero_division=0),
        f"{prefix}_precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0),
    }
    per_class = f1_score(y_true, y_pred, labels=list(range(len(classes))),
                         average=None, zero_division=0)
    for name, value in zip(classes, per_class):
        metrics[f"{prefix}_f1_{name}"] = value
    return {key: float(value) for key, value in metrics.items()}


def setup_mlflow(tracking_uri, experiment_name):
    """Atur tracking URI dan buat experiment (artefak di mlartifacts/)."""
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=(PROJECT_ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment_name)


def log_artifacts(y_test, pred, classes, features):
    """Simpan confusion matrix, classification report, dan daftar fitur."""
    fig = Figure(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, labels=list(range(len(classes))),
        display_labels=classes, ax=fig.subplots(), colorbar=False)
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix_test.png")
    mlflow.log_text(
        classification_report(y_test, pred,
                              labels=list(range(len(classes))),
                              target_names=classes, zero_division=0),
        "classification_report_test.txt")
    mlflow.log_text("\n".join(features), "features.txt")


def log_trained_model(model, args, signature, input_example):
    """Simpan model sebagai artefak MLflow sesuai flavor-nya."""
    if args.model == "xgboost":
        return mlflow.xgboost.log_model(
            xgb_model=model, name="model", signature=signature,
            input_example=input_example)
    return mlflow.sklearn.log_model(
        sk_model=model, name="model", signature=signature,
        input_example=input_example)


def run_training(args):
    """Jalankan satu eksperimen penuh dan kembalikan (run_id, metrik)."""
    classes = load_config()["model"]["classes"]
    data_path = Path(args.data)
    df, features = load_dataset(data_path, classes)
    train, test = time_split(df, args.test_size)
    x_train, y_train = train[features], train["target"]
    x_test, y_test = test[features], test["target"]
    logger.info("Data: %d baris (latih %d, uji %d), %d fitur",
                len(df), len(train), len(test), len(features))

    model = build_model(args)
    setup_mlflow(args.tracking_uri, args.experiment_name)
    commit, dirty = git_info()

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({
            **model_params(args),
            "seed": args.seed, "test_size": args.test_size,
            "cv_splits": args.cv_splits, "n_features": len(features),
            "n_train": len(train), "n_test": len(test),
            "data_rows": len(df), "data_start": str(df["timestamp"].min()),
            "data_end": str(df["timestamp"].max()),
            "data_md5": file_md5(data_path),
        })
        mlflow.set_tags({
            "task": "multiclass_classification",
            "data_version": args.data_version,
            "data_path": str(data_path),
            "git_commit": commit, "git_dirty": str(dirty),
            "label_mapping": ",".join(
                f"{i}={name}" for i, name in enumerate(classes)),
        })

        cv_mean, cv_std = walk_forward_cv(model, args, x_train, y_train,
                                          len(classes))
        fit_model(model, args, x_train, y_train)
        pred = model.predict(x_test)

        metrics = {"cv_f1_macro_mean": cv_mean, "cv_f1_macro_std": cv_std}
        metrics.update(compute_metrics(
            y_train, model.predict(x_train), classes, "train"))
        metrics.update(compute_metrics(y_test, pred, classes, "test"))
        mlflow.log_metrics(metrics)

        log_artifacts(y_test, pred, classes, features)
        info = log_trained_model(model, args, infer_signature(x_test, pred),
                                 x_train.head(5))
        mlflow.set_tag("model_uri", info.model_uri)

    logger.info("Run %s selesai | cv_f1_macro=%.4f | test_f1_macro=%.4f | "
                "test_accuracy=%.4f", args.run_name or run.info.run_id,
                cv_mean, metrics["test_f1_macro"], metrics["test_accuracy"])
    return run.info.run_id, metrics


def parse_args(argv=None):
    """Parse argumen CLI (semua punya nilai default)."""
    parser = argparse.ArgumentParser(
        description="Training klasifikasi tren harga + MLflow tracking.")
    parser.add_argument("--model", choices=MODELS, default="xgboost")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--c", type=float, default=1.0,
                        help="regularisasi logistic regression")
    parser.add_argument("--class-weight", choices=("none", "balanced"),
                        default="none")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--data-version", default="unversioned",
                        help="tag Git versi data, mis. data-v1.1.0")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--tracking-uri", default=default_tracking_uri())
    parser.add_argument("--run-name", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point. Mengembalikan exit code (0 = sukses, 1 = gagal)."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    try:
        run_training(args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
