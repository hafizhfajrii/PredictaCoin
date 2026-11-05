import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from train import DEFAULT_EXPERIMENT, default_tracking_uri

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "best_model.json"

logger = logging.getLogger("register_best")


def select_best_run(experiment_name, metric):
    """Ambil run non-dummy dengan nilai `metric` tertinggi (pandas Series)."""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="params.model != 'dummy'",
        order_by=[f"metrics.{metric} DESC"], max_results=100)
    column = f"metrics.{metric}"
    if runs.empty or column not in runs or "tags.model_uri" not in runs:
        raise ValueError(
            f"Belum ada run yang bisa dipilih di experiment "
            f"'{experiment_name}'. Jalankan train.py dulu.")
    runs = runs.dropna(subset=[column, "tags.model_uri"])
    if runs.empty:
        raise ValueError("Tidak ada run dengan metrik dan model lengkap.")
    return runs.iloc[0]


def collect(best, prefix):
    """Kumpulkan kolom `prefix.*` yang terisi menjadi dict."""
    return {col[len(prefix) + 1:]: best[col] for col in best.index
            if col.startswith(prefix + ".") and pd.notna(best[col])}


def build_metadata(best, metric, experiment_name, version, args):
    """Susun metadata model terbaik untuk tahap deployment."""
    params = collect(best, "params")
    tags = collect(best, "tags")
    return {
        "registered_model": {"name": args.model_name,
                             "version": str(version), "alias": args.alias},
        "selection": {"experiment": experiment_name, "metric": metric,
                      "value": float(best[f"metrics.{metric}"])},
        "run": {"run_id": best["run_id"],
                "run_name": tags.get("mlflow.runName")},
        "params": params,
        "metrics": {k: float(v) for k, v in collect(best, "metrics").items()},
        "data": {"version": tags.get("data_version"),
                 "md5": params.get("data_md5"),
                 "rows": params.get("data_rows"),
                 "start": params.get("data_start"),
                 "end": params.get("data_end")},
        "code": {"git_commit": tags.get("git_commit")},
        "registered_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
    }


def parse_args(argv=None):
    """Parse argumen CLI."""
    parser = argparse.ArgumentParser(
        description="Daftarkan run MLflow terbaik ke Model Registry.")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--metric", default="cv_f1_macro_mean",
                        help="metrik pemilihan (makin besar makin baik)")
    parser.add_argument(
        "--model-name", default="predictacoin-trend-classifier")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--tracking-uri", default=default_tracking_uri())
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point. Mengembalikan exit code (0 = sukses, 1 = gagal)."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    mlflow.set_tracking_uri(args.tracking_uri)
    try:
        best = select_best_run(args.experiment_name, args.metric)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    version = mlflow.register_model(best["tags.model_uri"],
                                    args.model_name).version
    MlflowClient().set_registered_model_alias(
        args.model_name, args.alias, version)

    metadata = build_metadata(best, args.metric, args.experiment_name,
                              version, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Run terbaik: %s (%s=%.4f)", metadata["run"]["run_name"],
                args.metric, metadata["selection"]["value"])
    logger.info("Terdaftar: %s versi %s (alias %s)", args.model_name,
                version, args.alias)
    logger.info("Metadata -> %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
