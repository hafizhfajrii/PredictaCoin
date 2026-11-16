import argparse
import logging
import os
import sys
import time
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from registry_tool import get_run
from train import DEFAULT_EXPERIMENT, default_tracking_uri, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLDS = {
    "min_f1_gain_over_baseline": 0.05,
    "min_class_f1": 0.25,
    "max_train_test_gap": 0.50,
    "max_latency_seconds": 5.0,
}
LATENCY_ROWS = 100

logger = logging.getLogger("validate_model")


def load_thresholds(cfg):
    """Gabungkan ambang batas default dengan bagian `validation` pada config."""
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(cfg.get("validation", {}))
    return thresholds


def make_check(name, value, op, limit, blocking=True):
    """Bangun satu hasil pemeriksaan (dict)."""
    passed = value >= limit if op == ">=" else value <= limit
    return {"name": name, "value": float(value), "op": op,
            "limit": float(limit), "passed": bool(passed), "blocking": blocking}


def evaluate_thresholds(metrics, baseline, latency, thresholds, classes):
    """Periksa metrik model terhadap baseline dan ambang batas.

    Args:
        metrics: metrik run model (dict).
        baseline: metrik run baseline dummy (dict).
        latency: waktu inference sebuah batch (detik).
        thresholds: ambang batas dari config.
        classes: daftar nama kelas, mis. ["Naik", "Turun", "Stabil"].

    Returns:
        (daftar pemeriksaan, True jika semua pemeriksaan wajib lolos).
    """
    needed = ["test_f1_macro", "train_f1_macro", "test_accuracy"]
    needed += [f"test_f1_{name}" for name in classes]
    missing = [m for m in needed if m not in metrics]
    missing += [f"baseline:{m}" for m in ("test_f1_macro", "test_accuracy")
                if m not in baseline]
    if missing:
        raise ValueError(f"Metrik wajib tidak ada: {missing}")

    class_f1 = min(metrics[f"test_f1_{name}"] for name in classes)
    checks = [
        make_check("test_f1_macro", metrics["test_f1_macro"], ">=",
                   baseline["test_f1_macro"]
                   + thresholds["min_f1_gain_over_baseline"]),
        make_check("min_class_f1", class_f1, ">=", thresholds["min_class_f1"]),
        make_check("gap_train_test",
                   metrics["train_f1_macro"] - metrics["test_f1_macro"], "<=",
                   thresholds["max_train_test_gap"]),
        make_check("latency_seconds", latency, "<=",
                   thresholds["max_latency_seconds"]),
        make_check("test_accuracy", metrics["test_accuracy"], ">=",
                   baseline["test_accuracy"], blocking=False),
    ]
    return checks, all(c["passed"] for c in checks if c["blocking"])


def status_label(check):
    """Label hasil: LULUS, GAGAL, atau INFO untuk pemeriksaan tidak wajib."""
    if check["passed"]:
        return "LULUS"
    return "GAGAL" if check["blocking"] else "INFO (tidak wajib)"


def measure_latency(model_uri, data_path, n_rows=LATENCY_ROWS):
    """Muat model dari registry dan ukur waktu prediksi satu batch (detik)."""
    path = Path(data_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Data untuk uji latensi tidak ditemukan: {path}")

    model = mlflow.pyfunc.load_model(model_uri)
    features = model.metadata.get_input_schema().input_names()
    batch = pd.read_parquet(path).dropna(subset=features).tail(n_rows)
    if batch.empty:
        raise ValueError("Tidak ada baris dengan fitur lengkap untuk uji latensi.")

    started = time.perf_counter()
    model.predict(batch[features].astype("float64"))
    return time.perf_counter() - started


def to_markdown(run_name, checks, passed):
    """Ringkasan validasi dalam format Markdown (untuk GitHub Step Summary)."""
    lines = [f"### Validasi Model (run `{run_name}`)", "",
             "| Pemeriksaan | Nilai | Syarat | Hasil |", "|---|---|---|---|"]
    for check in checks:
        lines.append(f"| {check['name']} | {check['value']:.4f} | "
                     f"{check['op']} {check['limit']:.4f} | "
                     f"{status_label(check)} |")
    verdict = ("LULUS: model boleh didaftarkan ke Model Registry (Staging)."
               if passed else
               "GAGAL: model TIDAK didaftarkan ke Model Registry.")
    lines += ["", f"**{verdict}**", ""]
    return "\n".join(lines)


def parse_args(argv=None):
    """Parse argumen CLI."""
    parser = argparse.ArgumentParser(
        description="Validasi metrik model terhadap baseline dan ambang batas.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--baseline-run-name", default=None,
                        help="default: <run-name>-baseline")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--tracking-uri", default=default_tracking_uri())
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point. Mengembalikan exit code (0 = lolos, 1 = gagal)."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    cfg = load_config()
    try:
        run = get_run(client, args.experiment_name, args.run_name)
        baseline = get_run(client, args.experiment_name,
                           args.baseline_run_name or f"{args.run_name}-baseline")
        latency = measure_latency(run.data.tags["model_uri"],
                                  run.data.tags["data_path"])
        checks, passed = evaluate_thresholds(
            run.data.metrics, baseline.data.metrics, latency,
            load_thresholds(cfg), cfg["model"]["classes"])
    except (ValueError, FileNotFoundError, KeyError, MlflowException) as exc:
        logger.error("%s", exc)
        return 1

    client.set_tag(run.info.run_id, "validation",
                   "passed" if passed else "failed")
    for check in checks:
        logger.info("%-16s %.4f %s %.4f -> %s", check["name"], check["value"],
                    check["op"], check["limit"], status_label(check))
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(Path(summary), "a", encoding="utf-8") as f:
            f.write(to_markdown(args.run_name, checks, passed))
    logger.info("Hasil validasi: %s", "LULUS" if passed else "GAGAL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
