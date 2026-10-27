"""
Preprocessing data untuk PredictaCoin (tahap Transform 1).

Data PredictaCoin adalah time series numerik (harga, market cap, volume),
sehingga preprocessing berfokus pada kualitas deret waktu, bukan teks
(tokenization / stopword removal tidak relevan).

Input : data/raw/{coin}/*.parquet
Output: data/interim/{coin}.parquet

Langkah:
1. Gabung semua partisi bulanan, validasi kolom wajib
2. Seragamkan timestamp (UTC, jam penuh, presisi ns), paksa kolom numerik,
   hapus duplikat (coin_id, timestamp) dengan mempertahankan data terbaru
3. Nilai tidak valid (inf, harga <= 0, market cap/volume negatif) -> missing
4. Deteksi gap waktu -> reindex ke deret jam lengkap, isi terbatas
   (ffill max 3 jam), tandai dengan is_gap_filled; gap lebih panjang dibuang
5. Flag outlier (robust z-score log return per jam) -> TIDAK dibuang, karena
   crash/pump adalah sinyal yang ingin dideteksi
6. Validasi skema dan rentang dengan pandera

Jalankan:
    python src/data/preprocess.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import pandera.pandas as pa
except ImportError:  # versi pandera lama
    import pandera as pa

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Nilai default; bisa dioverride lewat bagian "preprocessing" di config.yaml
MAX_FFILL_HOURS = 3
OUTLIER_WINDOW = 168      # 7 hari
OUTLIER_Z = 5.0
NUM_COLS = ["price_usd", "market_cap_usd", "total_volume_usd"]
RAW_COLUMNS = {"timestamp", "coin_id", "fetched_at", *NUM_COLS}

logger = logging.getLogger("preprocess")


def load_config():
    """Baca config/config.yaml dan kembalikan seluruh isinya sebagai dict."""
    with open(PROJECT_ROOT / "config" / "config.yaml",
              encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_params(cfg):
    """Ambil parameter preprocessing dari config, fallback ke default."""
    section = cfg.get("preprocessing", {})
    return {
        "max_ffill_hours": section.get("max_ffill_hours", MAX_FFILL_HOURS),
        "outlier_window": section.get("outlier_window_hours", OUTLIER_WINDOW),
        "outlier_z": section.get("outlier_z_threshold", OUTLIER_Z),
    }


def build_schema(coin):
    """Skema pandera untuk data interim satu coin."""
    return pa.DataFrameSchema(
        {
            "timestamp": pa.Column("datetime64[ns, UTC]", unique=True),
            "coin_id": pa.Column(checks=pa.Check.isin([coin])),
            "price_usd": pa.Column(float, pa.Check.gt(0)),
            "market_cap_usd": pa.Column(float, pa.Check.ge(0)),
            "total_volume_usd": pa.Column(float, pa.Check.ge(0)),
            "is_gap_filled": pa.Column(bool),
            "is_outlier": pa.Column(bool),
        },
        strict=True,
    )


def flag_outliers(price, window=OUTLIER_WINDOW, z_threshold=OUTLIER_Z):
    """Robust z-score (median/MAD) dari log return 1 jam, jendela 7 hari."""
    ret = np.log(price).diff()
    med = ret.rolling(window, min_periods=24).median()
    mad = (ret - med).abs().rolling(window, min_periods=24).median()
    z = 0.6745 * (ret - med) / mad.replace(0, np.nan)
    return (z.abs() > z_threshold).fillna(False)


def load_raw(coin, raw_dir):
    """Gabungkan semua partisi bulanan raw satu coin dan cek kolomnya."""
    files = sorted((raw_dir / coin).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"Tidak ada data raw untuk {coin}. Jalankan ingest_data.py dulu.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    missing = RAW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{coin}: kolom raw hilang: {sorted(missing)}")
    return df


def clean_coin(coin, raw_dir, params):
    """Bersihkan data satu coin dan kembalikan DataFrame interim."""
    df = load_raw(coin, raw_dir)
    n_raw = len(df)

    # 1-2. Seragamkan timestamp & tipe, hapus duplikat (data terbaru menang)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    df[NUM_COLS] = (df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan))
    df = (df.sort_values(["timestamp", "fetched_at"])
            .drop_duplicates(subset=["coin_id", "timestamp"], keep="last")
            .set_index("timestamp")[NUM_COLS])
    n_dedup = len(df)

    # 3. Nilai di luar rentang wajar dianggap missing (lalu ditangani gap)
    invalid = ((df["price_usd"] <= 0)
               | (df["market_cap_usd"] < 0)
               | (df["total_volume_usd"] < 0))
    n_invalid = int(invalid.sum())
    df.loc[invalid, NUM_COLS] = np.nan

    # 4. Gap: reindex ke deret jam lengkap, isi terbatas
    full_index = pd.date_range(
        df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full_index)
    df.index.name = "timestamp"
    gap_mask = df[NUM_COLS].isna().any(axis=1)
    df[NUM_COLS] = df[NUM_COLS].ffill(limit=params["max_ffill_hours"])
    df["is_gap_filled"] = gap_mask & df[NUM_COLS].notna().all(axis=1)
    n_unfilled = int(df[NUM_COLS].isna().any(axis=1).sum())
    df = df.dropna(subset=NUM_COLS)  # gap panjang tidak diisi, baris dibuang

    # 5. Flag outlier (tidak dibuang)
    df["is_outlier"] = flag_outliers(
        df["price_usd"], params["outlier_window"], params["outlier_z"])

    df = df.reset_index()
    df.insert(1, "coin_id", coin)
    df[NUM_COLS] = df[NUM_COLS].astype(float)

    # 6. Validasi
    build_schema(coin).validate(df)

    logger.info(
        "%s: raw=%d -> dedup=%d -> final=%d | tidak valid=%d, "
        "gap terisi=%d, gap tak terisi=%d, outlier=%d",
        coin, n_raw, n_dedup, len(df), n_invalid,
        int(df["is_gap_filled"].sum()), n_unfilled,
        int(df["is_outlier"].sum()))
    return df


def main():
    """Entry point. Mengembalikan exit code (0 = sukses, 1 = ada yang gagal)."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    cfg = load_config()
    data_cfg = cfg["data"]
    params = load_params(cfg)
    raw_dir = PROJECT_ROOT / data_cfg["raw_path"]
    out_dir = PROJECT_ROOT / data_cfg["interim_path"]
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for coin in data_cfg["assets"]:
        try:
            df = clean_coin(coin, raw_dir, params)
        except Exception:  # satu coin gagal tidak menghentikan coin lain
            logger.exception("Preprocessing gagal untuk %s", coin)
            failures.append(coin)
            continue
        path = out_dir / f"{coin}.parquet"
        df.to_parquet(path, index=False)
        logger.info("  -> %s", path.relative_to(PROJECT_ROOT))

    if failures:
        logger.error("Preprocessing gagal untuk: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
