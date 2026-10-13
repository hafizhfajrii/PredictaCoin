"""
Cleaning data untuk PredictaCoin (tahap Transform 1).

Input : data/raw/{coin}/*.parquet
Output: data/interim/{coin}.parquet

Langkah:
1. Gabung semua partisi bulanan, hapus duplikat (coin_id, timestamp)
2. Seragamkan timestamp (UTC, jam penuh, presisi ns)
3. Deteksi gap waktu -> reindex ke deret jam lengkap, isi terbatas (ffill max 3 jam),
   tandai dengan is_gap_filled
4. Flag outlier (robust z-score return per jam) -> TIDAK dibuang, karena crash/pump
   adalah sinyal yang ingin dideteksi
5. Validasi skema dan rentang dengan pandera

Jalankan:
    python src/data/clean.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import pandera.pandas as pa
except ImportError:  # versi pandera lama
    import pandera as pa

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAX_FFILL_HOURS = 3
OUTLIER_WINDOW = 168      # 7 hari
# ambang robust z-score (kandidat dipindah ke config.yaml)
OUTLIER_Z = 5.0
NUM_COLS = ["price_usd", "market_cap_usd", "total_volume_usd"]


def load_config():
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_schema(coin):
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


def flag_outliers(price):
    """Robust z-score (median/MAD) dari log return 1 jam, jendela 7 hari."""
    ret = np.log(price).diff()
    med = ret.rolling(OUTLIER_WINDOW, min_periods=24).median()
    mad = (ret - med).abs().rolling(OUTLIER_WINDOW, min_periods=24).median()
    z = 0.6745 * (ret - med) / mad.replace(0, np.nan)
    return (z.abs() > OUTLIER_Z).fillna(False)


def clean_coin(coin, raw_dir):
    files = sorted((raw_dir / coin).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"Tidak ada data raw untuk {coin}. Jalankan ingest.py dulu.")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    n_raw = len(df)

    # 1-2. Seragamkan timestamp, hapus duplikat
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    df = (df.sort_values(["timestamp", "fetched_at"])
            .drop_duplicates(subset=["coin_id", "timestamp"], keep="last")
            .set_index("timestamp")[NUM_COLS])
    n_dedup = len(df)

    # 3. Gap: reindex ke deret jam lengkap
    full_index = pd.date_range(
        df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full_index)
    df.index.name = "timestamp"
    gap_mask = df["price_usd"].isna()
    df[NUM_COLS] = df[NUM_COLS].ffill(limit=MAX_FFILL_HOURS)
    df["is_gap_filled"] = gap_mask & df["price_usd"].notna()
    n_unfilled = int(df["price_usd"].isna().sum())
    df = df.dropna(subset=NUM_COLS)  # gap > 3 jam tidak diisi, baris dibuang

    # 4. Flag outlier (tidak dibuang)
    df["is_outlier"] = flag_outliers(df["price_usd"])

    df = df.reset_index()
    df.insert(1, "coin_id", coin)
    df[NUM_COLS] = df[NUM_COLS].astype(float)

    # 5. Validasi
    build_schema(coin).validate(df)

    print(f"{coin}: raw={n_raw} -> dedup={n_dedup} -> final={len(df)} | "
          f"gap terisi={int(df['is_gap_filled'].sum())}, gap tak terisi={n_unfilled}, "
          f"outlier={int(df['is_outlier'].sum())}")
    return df


def main():
    cfg = load_config()["data"]
    raw_dir = PROJECT_ROOT / cfg["raw_path"]
    out_dir = PROJECT_ROOT / cfg["interim_path"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for coin in cfg["assets"]:
        df = clean_coin(coin, raw_dir)
        path = out_dir / f"{coin}.parquet"
        df.to_parquet(path, index=False)
        print(f"  -> {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
