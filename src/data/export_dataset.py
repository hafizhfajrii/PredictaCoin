"""
Ekspor dataset mentah gabungan untuk versioning DVC (LK-05).

Menggabungkan seluruh partisi data/raw/{coin}/*menjadi satu file data/dvc/dataset_raw.parquet. Duplikat
(coin_id, timestamp) dibuang dengan mempertahankan data terbaru.

"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "dvc" / "dataset_raw.parquet"

logger = logging.getLogger("export_dataset")


def load_config():
    """Baca config/config.yaml dan kembalikan seluruh isinya sebagai dict."""
    with open(PROJECT_ROOT / "config" / "config.yaml",
              encoding="utf-8") as f:
        return yaml.safe_load(f)


def export_dataset(raw_dir, assets, output_path=OUTPUT_PATH):
    """Gabungkan semua partisi raw semua coin menjadi satu file parquet.

    Args:
        raw_dir: folder data/raw.
        assets: daftar coin_id, mis. ["bitcoin", "ethereum"].
        output_path: lokasi file keluaran.

    Returns:
        DataFrame gabungan yang sudah di-dedup.

    Raises:
        FileNotFoundError: jika salah satu coin belum punya data raw.
    """
    frames = []
    for coin in assets:
        files = sorted((raw_dir / coin).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"Tidak ada data raw untuk {coin}. "
                "Jalankan ingest_data.py dulu.")
        frames.extend(pd.read_parquet(path) for path in files)

    df = pd.concat(frames, ignore_index=True)
    df = (df.sort_values(["coin_id", "timestamp", "fetched_at"])
            .drop_duplicates(subset=["coin_id", "timestamp"], keep="last")
            .reset_index(drop=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def main():
    """Entry point. Mengembalikan exit code (0 = sukses, 1 = gagal)."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    cfg = load_config()["data"]
    raw_dir = PROJECT_ROOT / cfg["raw_path"]

    try:
        df = export_dataset(raw_dir, cfg["assets"])
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Dataset diekspor -> %s",
                OUTPUT_PATH.relative_to(PROJECT_ROOT))
    logger.info("Total baris: %d", len(df))
    for coin, part in df.groupby("coin_id"):
        logger.info("  %s: %d baris, %s s/d %s", coin, len(part),
                    part["timestamp"].min(), part["timestamp"].max())
    return 0


if __name__ == "__main__":
    sys.exit(main())
