"""
Entry point pipeline data PredictaCoin:
ingest_data -> preprocess -> build_features.

Jika salah satu tahap gagal, pipeline berhenti dengan exit code 1 sehingga
tidak ada data setengah jadi yang di-commit.

Jalankan:
    python src/data/run_pipeline.py
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "0.1.0"

STEPS = [
    ("1. Ingestion", "src/data/ingest_data.py"),
    ("2. Preprocessing", "src/data/preprocess.py"),
    ("3. Feature engineering", "src/features/build_features.py"),
]

logger = logging.getLogger("run_pipeline")


def git_commit():
    """Kembalikan hash commit singkat kode saat ini ("unknown" jika gagal)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=PROJECT_ROOT, capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def write_metadata():
    """Tulis data/processed/metadata.json dari features.parquet."""
    df = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "features.parquet")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "git_commit_code": git_commit(),
        "rows": int(len(df)),
        "coins": sorted(df["coin_id"].unique().tolist()),
        "time_range_utc": [df["timestamp"].min().isoformat(),
                           df["timestamp"].max().isoformat()],
        "label_distribution": df["label"].value_counts(
            normalize=True).round(3).to_dict(),
    }
    path = PROJECT_ROOT / "data" / "processed" / "metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Metadata -> %s", path.relative_to(PROJECT_ROOT))


def main():
    """Jalankan semua tahap berurutan; berhenti di tahap pertama yang gagal."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    start = time.time()
    for name, script in STEPS:
        logger.info("===== %s (%s) =====", name, script)
        try:
            subprocess.run([sys.executable, str(PROJECT_ROOT / script)],
                           cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            logger.error("[GAGAL] %s berhenti dengan kode %d",
                         name, exc.returncode)
            sys.exit(1)
    write_metadata()
    logger.info("Pipeline selesai dalam %.1f detik", time.time() - start)


if __name__ == "__main__":
    main()
