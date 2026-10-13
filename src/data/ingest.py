"""
Ingestion data CoinGecko untuk PredictaCoin (tahap Extract).

Backfill 90 hari saat data belum ada, selanjutnya incremental (2 hari).
Jalankan:
    python src/data/ingest.py
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

BASE_URL = "https://api.coingecko.com/api/v3"
MAX_RETRIES = 5


def load_config():
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_market_chart(coin_id, days, api_key):
    """Panggil /market_chart dengan retry + exponential backoff (429 / 5xx)."""
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    headers = {"x-cg-demo-api-key": api_key}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params,
                                headers=headers, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(
                    f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException,) as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt * 5  # 10, 20, 40, 80 detik
            print(
                f"[retry {attempt}/{MAX_RETRIES}] {coin_id}: {exc}. Tunggu {wait}s")
            time.sleep(wait)


def to_dataframe(data, coin_id, fetched_at):
    """Ubah JSON CoinGecko menjadi DataFrame sesuai skema raw."""
    df_p = pd.DataFrame(data.get("prices", []), columns=["ts", "price_usd"])
    df_v = pd.DataFrame(data.get("total_volumes", []),
                        columns=["ts", "total_volume_usd"])
    df_c = pd.DataFrame(data.get("market_caps", []),
                        columns=["ts", "market_cap_usd"])

    df = df_p.merge(df_v, on="ts").merge(df_c, on="ts")
    df["timestamp"] = pd.to_datetime(
        df["ts"], unit="ms", utc=True).dt.floor("h")
    df["coin_id"] = coin_id
    df["fetched_at"] = fetched_at
    df = df[["timestamp", "coin_id", "price_usd", "market_cap_usd",
             "total_volume_usd", "fetched_at"]]

    # Titik terakhir biasanya jam "setengah-jalan" dan bisa berubah di fetch
    # berikutnya. Setelah dibulatkan, simpan satu baris per jam (yang terbaru).
    return df.drop_duplicates(subset=["coin_id", "timestamp"], keep="last")


def save_partitioned(df, coin_id, raw_dir):
    """Append ke file bulanan; baris baru menimpa baris lama untuk jam yang sama."""
    coin_dir = raw_dir / coin_id
    coin_dir.mkdir(parents=True, exist_ok=True)

    df = df.assign(month=df["timestamp"].dt.strftime("%Y-%m"))
    for month, part in df.groupby("month"):
        part = part.drop(columns="month")
        path = coin_dir / f"{month}.parquet"
        if path.exists():
            part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
        part = (part.drop_duplicates(subset=["coin_id", "timestamp"], keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True))
        part.to_parquet(path, index=False)
        print(f"  -> {path.relative_to(PROJECT_ROOT)} ({len(part)} baris)")


def main():
    cfg = load_config()["data"]
    api_key = os.getenv("COINGECKO_API_KEY")
    if not api_key:
        raise ValueError(
            "COINGECKO_API_KEY tidak ditemukan (.env atau GitHub Secret).")

    raw_dir = PROJECT_ROOT / cfg["raw_path"]
    fetched_at = datetime.now(timezone.utc)

    for coin in cfg["assets"]:
        coin_dir = raw_dir / coin
        is_first_run = not (coin_dir.exists() and any(
            coin_dir.glob("*.parquet")))
        days = cfg["history_window_days"] if is_first_run else cfg["incremental_days"]
        mode = "BACKFILL" if is_first_run else "INCREMENTAL"
        print(f"[{mode}] {coin}: days={days}")

        data = fetch_market_chart(coin, days, api_key)
        df = to_dataframe(data, coin, fetched_at)
        print(f"  status OK, {len(df)} titik, "
              f"{df['timestamp'].min()} s/d {df['timestamp'].max()}")
        save_partitioned(df, coin, raw_dir)
        time.sleep(2)  # jeda sopan antar request


if __name__ == "__main__":
    main()
