"""
Feature engineering untuk PredictaCoin (tahap Transform 2).

Input : data/interim/{coin}.parquet
Output: data/processed/features.parquet  (baris berlabel, untuk training)
        data/processed/latest.parquet    (baris terbaru tanpa label, untuk inference)

Jalankan:
    python src/features/build_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ta.momentum import RSIIndicator
from ta.trend import MACD

PROJECT_ROOT = Path(__file__).resolve().parents[2]

META_COLS = ["timestamp", "coin_id", "price_usd"]
TARGET_COLS = ["future_return_pct", "label"]


def load_config():
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_coin_features(df, horizon, up_pct, down_pct):
    df = df.sort_values("timestamp").reset_index(drop=True)
    price = df["price_usd"]
    volume = df["total_volume_usd"]
    log_price = np.log(price)
    ret1 = log_price.diff()

    out = df[META_COLS].copy()

    # Return
    out["ret_1h"] = ret1
    for h in (3, 6, 24):
        out[f"ret_{h}h"] = log_price.diff(h)

    # Rolling statistik return (volatilitas)
    for w in (6, 24):
        out[f"ret_mean_{w}h"] = ret1.rolling(w).mean()
        out[f"ret_std_{w}h"] = ret1.rolling(w).std()

    # Posisi harga terhadap rata-rata 24 jam
    out["price_vs_ma24"] = price / price.rolling(24).mean() - 1

    # Indikator teknikal
    out["rsi_14"] = RSIIndicator(price, window=14).rsi()
    macd = MACD(price)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    # Perubahan volume
    out["vol_chg_1h"] = volume.pct_change(1)
    out["vol_chg_24h"] = volume.pct_change(24)

    # Lag features
    for lag in (1, 2, 3):
        out[f"ret_1h_lag{lag}"] = ret1.shift(lag)

    # Fitur waktu
    out["hour"] = df["timestamp"].dt.hour
    out["dayofweek"] = df["timestamp"].dt.dayofweek

    # Flag outlier dari tahap cleaning (dihitung dari data masa lalu saja)
    out["is_outlier"] = df["is_outlier"].astype(int)

    # Label: perubahan harga horizon jam ke depan
    gap = df["timestamp"].shift(-horizon) - df["timestamp"]
    valid = gap == pd.Timedelta(hours=horizon)
    future_price = price.shift(-horizon).where(valid)
    future_ret = (future_price / price - 1) * 100
    label = pd.Series(
        np.select([future_ret > up_pct, future_ret < down_pct],
                  ["Naik", "Turun"], default="Stabil"),
        index=out.index,
    )
    out["future_return_pct"] = future_ret
    out["label"] = label.where(future_ret.notna())

    return out.replace([np.inf, -np.inf], np.nan)


def main():
    cfg = load_config()
    data_cfg, lab_cfg = cfg["data"], cfg["labeling"]
    interim_dir = PROJECT_ROOT / data_cfg["interim_path"]
    out_dir = PROJECT_ROOT / data_cfg["processed_path"]
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for coin in data_cfg["assets"]:
        path = interim_dir / f"{coin}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} tidak ada. Jalankan clean.py dulu.")
        frames.append(build_coin_features(
            pd.read_parquet(path),
            horizon=lab_cfg["horizon_hours"],
            up_pct=lab_cfg["threshold_up_pct"],
            down_pct=lab_cfg["threshold_down_pct"],
        ))
    all_df = pd.concat(frames, ignore_index=True)

    feature_cols = [
        c for c in all_df.columns if c not in META_COLS + TARGET_COLS]
    # buang baris warm-up (rolling/RSI/MACD)
    all_df = all_df.dropna(subset=feature_cols)

    labeled = all_df.dropna(subset=["label"]).reset_index(drop=True)
    latest = (all_df[all_df["label"].isna()]
              .sort_values("timestamp").groupby("coin_id").tail(1)
              .reset_index(drop=True))

    labeled.to_parquet(out_dir / "features.parquet", index=False)
    latest.to_parquet(out_dir / "latest.parquet", index=False)

    print(f"Fitur ({len(feature_cols)}): {feature_cols}")
    print(
        f"features.parquet: {len(labeled)} baris | latest.parquet: {len(latest)} baris")
    print("\nDistribusi label (proporsi) per coin:")
    print(labeled.groupby("coin_id")[
          "label"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
