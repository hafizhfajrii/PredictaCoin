"""
Modul pengambilan data CoinGecko API untuk PredictaCoin.

Cara pakai cepat:
    from src.data.fetch_coingecko import fetch_current_price, fetch_historical

    fetch_current_price()
    df = fetch_historical_as_dataframe("bitcoin", days=7)
"""

import os
import json
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

# Cari file .env di root project (dua level di atas file ini: src/data/ -> root),
# bukan bergantung pada working directory saat dipanggil dari notebook/script lain.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

API_KEY = os.getenv("COINGECKO_API_KEY")
BASE_URL = "https://api.coingecko.com/api/v3"


def _check_api_key():
    if not API_KEY:
        raise ValueError(
            "COINGECKO_API_KEY tidak ditemukan. "
            "Salin .env.example menjadi .env dan isi dengan API key kamu."
        )


def fetch_current_price(coins=("bitcoin", "ethereum")):
    """Ambil harga real-time untuk daftar coin yang diberikan."""
    _check_api_key()
    url = f"{BASE_URL}/simple/price"
    params = {
        "ids": ",".join(coins),
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_last_updated_at": "true",
        "x_cg_demo_api_key": API_KEY,
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    print(f"Status code: {response.status_code}")
    print(f"Waktu fetch: {datetime.now().isoformat()}")
    print(json.dumps(data, indent=2))
    return data


def fetch_historical(coin_id="bitcoin", days=7):
    """Ambil data historis harga (raw JSON) untuk satu coin."""
    _check_api_key()
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "x_cg_demo_api_key": API_KEY,
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_historical_as_dataframe(coin_id="bitcoin", days=7):
    """
    Ambil data historis dan langsung konversi ke pandas DataFrame,
    siap dipakai untuk EDA atau feature engineering.

    Kolom: timestamp, price, volume, market_cap
    """
    data = fetch_historical(coin_id, days)

    df_price = pd.DataFrame(data.get("prices", []), columns=["ts", "price"])
    df_volume = pd.DataFrame(
        data.get("total_volumes", []), columns=["ts", "volume"])
    df_cap = pd.DataFrame(data.get("market_caps", []),
                          columns=["ts", "market_cap"])

    df = df_price.merge(df_volume, on="ts").merge(df_cap, on="ts")
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
    df["coin_id"] = coin_id
    df = df[["timestamp", "coin_id", "price", "volume", "market_cap"]]

    return df


def save_raw_data(df, coin_id, out_dir=None):
    """Simpan DataFrame ke data/raw sebagai CSV dengan nama berbasis tanggal fetch."""
    if out_dir is None:
        out_dir = str(_PROJECT_ROOT / "data" / "raw")
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{coin_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"Data disimpan ke: {path}")
    return path


if __name__ == "__main__":
    print("=== Fetch harga real-time ===")
    fetch_current_price()

    print("\n=== Fetch data historis (untuk training awal) ===")
    for coin in ["bitcoin", "ethereum"]:
        df = fetch_historical_as_dataframe(coin, days=7)
        print(f"\n{coin}: {len(df)} baris")
        print(df.head(3))
        save_raw_data(df, coin)
