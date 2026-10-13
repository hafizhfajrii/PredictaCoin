"""
Entry point pipeline data PredictaCoin: ingest -> clean -> build_features.

Jalankan:
    python src/data/run_pipeline.py
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "0.1.0"

STEPS = [
    ("1. Ingestion", "src/data/ingest.py"),
    ("2. Cleaning", "src/data/clean.py"),
    ("3. Feature engineering", "src/features/build_features.py"),
]


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def write_metadata():
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features.parquet")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit_code": git_commit(),
        "rows": int(len(df)),
        "coins": sorted(df["coin_id"].unique().tolist()),
        "time_range_utc": [df["timestamp"].min().isoformat(),
                           df["timestamp"].max().isoformat()],
        "label_distribution": df["label"].value_counts(normalize=True).round(3).to_dict(),
    }
    path = PROJECT_ROOT / "data" / "processed" / "metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nMetadata -> {path.relative_to(PROJECT_ROOT)}")


def main():
    start = time.time()
    for name, script in STEPS:
        print(f"\n===== {name} ({script}) =====")
        try:
            subprocess.run([sys.executable, str(PROJECT_ROOT / script)],
                           cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"\n[GAGAL] {name} berhenti dengan kode {exc.returncode}")
            sys.exit(1)
    write_metadata()
    print(f"\nPipeline selesai dalam {time.time() - start:.1f} detik")


if __name__ == "__main__":
    main()