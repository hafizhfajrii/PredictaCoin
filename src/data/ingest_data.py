"""
Data ingestion CoinGecko untuk PredictaCoin (tahap Extract).

Perilaku:
    * Run pertama (belum ada data)  -> BACKFILL (history_window_days, 90 hari).
    * Run berikutnya                -> INCREMENTAL (incremental_days, 2 hari).
    * Data lama tidak pernah ditimpa secara destruktif:
        - data/raw/{coin}/{YYYY-MM}.parquet di-append lalu di-dedup per
          (coin_id, timestamp), ditulis secara atomik (file sementara).
        - Respons JSON mentah disimpan sebagai snapshot ber-timestamp di
          data/raw/snapshots/{coin}_{YYYYMMDDTHHMMSSZ}.json.

Penanganan error:
    * Timeout / koneksi putus / HTTP 429 / HTTP 5xx / respons bukan JSON
      -> retry dengan exponential backoff (hormati header Retry-After).
    * HTTP 4xx lain (mis. 401, 404) -> gagal langsung (retry tidak berguna).
    * Respons kosong atau tidak lengkap -> ditolak sebelum disimpan.
    * Satu coin gagal tidak menghentikan coin lain; exit code 1 di akhir.

Jalankan:
    python src/data/ingest_data.py
    python src/data/ingest_data.py --coins bitcoin --days 2
    python src/data/ingest_data.py --no-snapshot
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

BASE_URL = "https://api.coingecko.com/api/v3"
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 5          # tunggu 10, 20, 40, 80 detik
MAX_WAIT_SECONDS = 120
REQUEST_TIMEOUT_SECONDS = 30
POLITE_DELAY_SECONDS = 2          # jeda sopan antar coin
REQUIRED_KEYS = ("prices", "market_caps", "total_volumes")
SNAPSHOT_TS_FORMAT = "%Y%m%dT%H%M%SZ"
SNAPSHOT_PATTERN = re.compile(r"^.+_(?P<ts>\d{8}T\d{6}Z)\.json$")

logger = logging.getLogger("ingest_data")


class IngestionError(Exception):
    """Kegagalan ingestion yang tidak bisa dipulihkan untuk satu coin."""


class FatalAPIError(IngestionError):
    """Error HTTP 4xx (selain 429): retry tidak akan membantu."""


def load_config():
    """Baca config/config.yaml dan kembalikan seluruh isinya sebagai dict."""
    with open(PROJECT_ROOT / "config" / "config.yaml",
              encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_retry_after(resp):
    """Ambil nilai header Retry-After (detik), atau None jika tidak ada."""
    try:
        return float(resp.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


def _backoff_seconds(attempt, retry_after=None):
    """Hitung lama tunggu: exponential, minimal Retry-After, maksimal 120 dtk."""
    wait = BACKOFF_BASE_SECONDS * 2 ** attempt
    if retry_after:
        wait = max(wait, retry_after)
    return min(wait, MAX_WAIT_SECONDS)


def fetch_market_chart(coin_id, days, api_key):
    """Panggil endpoint /market_chart dengan retry dan exponential backoff.

    Args:
        coin_id: id koin CoinGecko, mis. "bitcoin".
        days: rentang hari yang diminta (granularitas otomatis per jam).
        api_key: CoinGecko Demo API key.

    Returns:
        dict hasil parse JSON.

    Raises:
        FatalAPIError: HTTP 4xx selain 429.
        IngestionError: gagal setelah MAX_RETRIES percobaan.
    """
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    headers = {"x-cg-demo-api-key": api_key}

    for attempt in range(1, MAX_RETRIES + 1):
        retry_after = None
        try:
            resp = requests.get(url, params=params, headers=headers,
                                timeout=REQUEST_TIMEOUT_SECONDS)
        except (requests.ConnectionError, requests.Timeout) as exc:
            reason = type(exc).__name__
        except requests.RequestException as exc:
            raise IngestionError(
                f"{coin_id}: request tidak valid: {exc}") from exc
        else:
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    reason = "respons bukan JSON valid"
            elif resp.status_code == 429 or resp.status_code >= 500:
                reason = f"HTTP {resp.status_code}"
                retry_after = _parse_retry_after(resp)
            else:
                hint = ""
                if resp.status_code in (401, 403):
                    hint = " (cek COINGECKO_API_KEY)"
                elif resp.status_code == 404:
                    hint = " (cek coin_id di config.yaml)"
                raise FatalAPIError(
                    f"{coin_id}: HTTP {resp.status_code}{hint}")

        if attempt == MAX_RETRIES:
            raise IngestionError(
                f"{coin_id}: gagal setelah {MAX_RETRIES} percobaan ({reason})")
        wait = _backoff_seconds(attempt, retry_after)
        logger.warning("[retry %d/%d] %s: %s. Tunggu %.0f detik",
                       attempt, MAX_RETRIES, coin_id, reason, wait)
        time.sleep(wait)

    raise IngestionError(f"{coin_id}: gagal mengambil data")  # pragma: no cover


def validate_payload(data, coin_id):
    """Pastikan respons berisi prices, market_caps, total_volumes yang tidak kosong."""
    if not isinstance(data, dict):
        raise IngestionError(f"{coin_id}: format respons tidak dikenali")
    missing = [key for key in REQUIRED_KEYS if not data.get(key)]
    if missing:
        raise IngestionError(
            f"{coin_id}: respons kosong/tidak lengkap, kunci bermasalah: "
            f"{missing}")


def to_dataframe(data, coin_id, fetched_at):
    """Ubah JSON CoinGecko menjadi DataFrame sesuai skema raw.

    Timestamp dibulatkan ke jam penuh UTC. Titik terakhir dari API biasanya
    jam "setengah-jalan" yang bisa berubah di fetch berikutnya, jadi setelah
    pembulatan hanya satu baris per jam (yang terbaru) yang dipertahankan.
    """
    df_p = pd.DataFrame(data["prices"], columns=["ts", "price_usd"])
    df_v = pd.DataFrame(data["total_volumes"],
                        columns=["ts", "total_volume_usd"])
    df_c = pd.DataFrame(data["market_caps"],
                        columns=["ts", "market_cap_usd"])

    df = df_p.merge(df_v, on="ts").merge(df_c, on="ts")
    n_before = len(df)
    df = df.dropna(subset=["price_usd"])
    if len(df) < n_before:
        logger.warning("%s: %d titik tanpa harga dibuang",
                       coin_id, n_before - len(df))
    if df.empty:
        raise IngestionError(f"{coin_id}: tidak ada titik data yang valid")

    df["timestamp"] = pd.to_datetime(
        df["ts"], unit="ms", utc=True).dt.floor("h")
    df["coin_id"] = coin_id
    df["fetched_at"] = fetched_at
    df = df[["timestamp", "coin_id", "price_usd", "market_cap_usd",
             "total_volume_usd", "fetched_at"]]
    return df.drop_duplicates(subset=["coin_id", "timestamp"], keep="last")


def save_snapshot(data, coin_id, days, fetched_at, snapshot_dir):
    """Simpan respons JSON mentah ber-timestamp (tidak menimpa snapshot lama)."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    name = f"{coin_id}_{fetched_at.strftime(SNAPSHOT_TS_FORMAT)}.json"
    path = snapshot_dir / name
    payload = {
        "coin_id": coin_id,
        "days": days,
        "fetched_at": fetched_at.isoformat(timespec="seconds"),
        "data": data,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")),
                    encoding="utf-8")
    logger.info("  snapshot -> %s", path.relative_to(PROJECT_ROOT))
    return path


def prune_snapshots(snapshot_dir, retention_days, now):
    """Hapus snapshot yang lebih tua dari retention_days (0 = simpan semua).

    Hanya berdasarkan timestamp di nama file. Data utama (parquet bulanan)
    tidak disentuh, jadi tidak ada data yang hilang.
    """
    if retention_days <= 0 or not snapshot_dir.exists():
        return
    cutoff = now - timedelta(days=retention_days)
    removed = 0
    for path in snapshot_dir.glob("*.json"):
        match = SNAPSHOT_PATTERN.match(path.name)
        if not match:
            continue
        ts = datetime.strptime(match["ts"], SNAPSHOT_TS_FORMAT)
        if ts.replace(tzinfo=timezone.utc) < cutoff:
            path.unlink()
            removed += 1
    if removed:
        logger.info("Snapshot lama (> %d hari) dihapus: %d file",
                    retention_days, removed)


def save_partitioned(df, coin_id, raw_dir):
    """Append ke file bulanan; untuk jam yang sama baris terbaru menang.

    Penulisan atomik: ditulis ke file sementara lalu di-replace, sehingga
    file lama tidak rusak jika proses terhenti di tengah jalan.
    """
    coin_dir = raw_dir / coin_id
    coin_dir.mkdir(parents=True, exist_ok=True)

    df = df.assign(month=df["timestamp"].dt.strftime("%Y-%m"))
    for month, part in df.groupby("month"):
        part = part.drop(columns="month")
        path = coin_dir / f"{month}.parquet"
        if path.exists():
            part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
        part = (part.drop_duplicates(subset=["coin_id", "timestamp"],
                                     keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True))
        tmp_path = path.with_suffix(".parquet.tmp")
        part.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
        logger.info("  -> %s (%d baris)", path.relative_to(PROJECT_ROOT),
                    len(part))


def ingest_coin(coin, days, mode, api_key, fetched_at, raw_dir,
                snapshot_dir):
    """Jalankan seluruh tahap ingestion untuk satu coin."""
    logger.info("[%s] %s: days=%s", mode, coin, days)
    data = fetch_market_chart(coin, days, api_key)
    validate_payload(data, coin)
    df = to_dataframe(data, coin, fetched_at)
    logger.info("  status OK, %d titik, %s s/d %s",
                len(df), df["timestamp"].min(), df["timestamp"].max())
    if snapshot_dir is not None:
        save_snapshot(data, coin, days, fetched_at, snapshot_dir)
    save_partitioned(df, coin, raw_dir)


def parse_args(argv=None):
    """Parse argumen CLI (semuanya opsional, default dari config.yaml)."""
    parser = argparse.ArgumentParser(
        description="Ingestion data CoinGecko (backfill / incremental).")
    parser.add_argument("--coins", nargs="+",
                        help="override daftar coin (default: config.yaml)")
    parser.add_argument("--days", type=int,
                        help="override jumlah hari (default: otomatis)")
    parser.add_argument("--no-snapshot", action="store_true",
                        help="jangan simpan snapshot JSON mentah")
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point. Mengembalikan exit code (0 = sukses, 1 = ada yang gagal)."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    cfg = load_config()
    data_cfg = cfg["data"]
    ing_cfg = cfg.get("ingestion", {})

    api_key = os.getenv("COINGECKO_API_KEY")
    if not api_key:
        logger.error("COINGECKO_API_KEY tidak ditemukan "
                     "(.env atau GitHub Secret).")
        return 1

    raw_dir = PROJECT_ROOT / data_cfg["raw_path"]
    snapshot_dir = None
    if ing_cfg.get("snapshot_enabled", True) and not args.no_snapshot:
        snapshot_dir = PROJECT_ROOT / ing_cfg.get("snapshot_path",
                                                  "data/raw/snapshots")
    fetched_at = datetime.now(timezone.utc)
    coins = args.coins or data_cfg["assets"]

    failures = []
    for index, coin in enumerate(coins):
        coin_dir = raw_dir / coin
        is_first_run = not (coin_dir.exists()
                            and any(coin_dir.glob("*.parquet")))
        if args.days:
            days, mode = args.days, "MANUAL"
        elif is_first_run:
            days, mode = data_cfg["history_window_days"], "BACKFILL"
        else:
            days, mode = data_cfg["incremental_days"], "INCREMENTAL"

        try:
            ingest_coin(coin, days, mode, api_key, fetched_at, raw_dir,
                        snapshot_dir)
        except IngestionError as exc:
            logger.error("%s", exc)
            failures.append(coin)
        if index < len(coins) - 1:
            time.sleep(POLITE_DELAY_SECONDS)

    if snapshot_dir is not None:
        prune_snapshots(snapshot_dir,
                        ing_cfg.get("snapshot_retention_days", 7),
                        fetched_at)

    if failures:
        logger.error("Ingestion gagal untuk: %s", ", ".join(failures))
        return 1
    logger.info("Ingestion selesai untuk: %s", ", ".join(coins))
    return 0


if __name__ == "__main__":
    sys.exit(main())
