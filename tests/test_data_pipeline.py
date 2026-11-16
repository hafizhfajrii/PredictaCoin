"""Unit test tahap ingestion dan preprocessing (tanpa jaringan)."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import ingest_data
import preprocess

PARAMS = {"max_ffill_hours": 3, "outlier_window": 168, "outlier_z": 5.0}


class FakeResponse:
    """Pengganti requests.Response sederhana."""

    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("bukan JSON")
        return self._payload


def make_payload(n=5, start_ms=1_700_000_000_000):
    """Payload CoinGecko palsu: titik tiap 30 menit."""
    points = [[start_ms + i * 1_800_000, 100.0 + i] for i in range(n)]
    return {"prices": points,
            "market_caps": [[t, p * 10] for t, p in points],
            "total_volumes": [[t, p * 2] for t, p in points]}


# ----------------------------------------------------------------- ingestion
def test_to_dataframe_floors_to_hour_and_keeps_latest():
    df = ingest_data.to_dataframe(make_payload(), "bitcoin",
                                  datetime.now(timezone.utc))
    assert len(df) == 3
    assert (df["timestamp"].dt.minute == 0).all()
    assert df.iloc[0]["price_usd"] == 101.0  # titik terakhir pada jam yang sama
    assert not df.duplicated(["coin_id", "timestamp"]).any()


def test_validate_payload_rejects_empty_and_missing_keys():
    empty = make_payload()
    empty["prices"] = []
    with pytest.raises(ingest_data.IngestionError):
        ingest_data.validate_payload(empty, "bitcoin")
    with pytest.raises(ingest_data.IngestionError):
        ingest_data.validate_payload({"prices": [[1, 2]]}, "bitcoin")


def test_backoff_grows_respects_retry_after_and_is_capped():
    assert ingest_data._backoff_seconds(1) == 10
    assert ingest_data._backoff_seconds(1, retry_after=60) == 60
    assert ingest_data._backoff_seconds(5) == ingest_data.MAX_WAIT_SECONDS


def test_fetch_fails_fast_on_client_error(monkeypatch):
    calls = []
    monkeypatch.setattr(ingest_data.requests, "get",
                        lambda *a, **k: calls.append(1) or FakeResponse(404))
    with pytest.raises(ingest_data.FatalAPIError):
        ingest_data.fetch_market_chart("bitcoin", 2, "key")
    assert len(calls) == 1  # 4xx tidak di-retry


def test_fetch_retries_rate_limit_then_succeeds(monkeypatch):
    responses = iter([FakeResponse(429, headers={"Retry-After": "1"}),
                      FakeResponse(200, payload=make_payload())])
    monkeypatch.setattr(ingest_data.requests, "get",
                        lambda *a, **k: next(responses))
    monkeypatch.setattr(ingest_data.time, "sleep", lambda s: None)
    assert "prices" in ingest_data.fetch_market_chart("bitcoin", 2, "key")


def test_fetch_gives_up_after_max_retries(monkeypatch):
    calls = []
    monkeypatch.setattr(ingest_data.requests, "get",
                        lambda *a, **k: calls.append(1) or FakeResponse(500))
    monkeypatch.setattr(ingest_data.time, "sleep", lambda s: None)
    with pytest.raises(ingest_data.IngestionError):
        ingest_data.fetch_market_chart("bitcoin", 2, "key")
    assert len(calls) == ingest_data.MAX_RETRIES


# ------------------------------------------------------------- preprocessing
def write_raw(tmp_path, n=200, drop=(), invalid=None):
    """Tulis data raw sintetis (dengan satu baris duplikat) dan kembalikan folder."""
    index = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    df = pd.DataFrame({
        "timestamp": index, "coin_id": "bitcoin", "price_usd": price,
        "market_cap_usd": price * 1e6, "total_volume_usd": price * 1e3,
        "fetched_at": pd.Timestamp("2026-02-01", tz="UTC")})
    if invalid is not None:
        df.loc[invalid, "price_usd"] = 0.0
    df = df.drop(index=list(drop))
    df = pd.concat([df, df.iloc[[10]]])
    (tmp_path / "bitcoin").mkdir()
    df.to_parquet(tmp_path / "bitcoin" / "2026-01.parquet", index=False)
    return tmp_path


def test_clean_coin_dedups_and_fills_short_gap(tmp_path):
    out = preprocess.clean_coin("bitcoin", write_raw(tmp_path, drop=[50, 51]),
                                PARAMS)
    assert len(out) == 200
    assert out["is_gap_filled"].sum() == 2
    assert not out.duplicated(["coin_id", "timestamp"]).any()
    assert (out["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all()


def test_clean_coin_drops_gap_longer_than_ffill_limit(tmp_path):
    out = preprocess.clean_coin(
        "bitcoin", write_raw(tmp_path, drop=range(60, 65)), PARAMS)
    assert out["is_gap_filled"].sum() == 3  # hanya 3 jam pertama yang diisi
    assert len(out) == 198                  # 2 jam sisanya dibuang


def test_invalid_price_is_treated_as_gap(tmp_path):
    out = preprocess.clean_coin("bitcoin", write_raw(tmp_path, invalid=80),
                                PARAMS)
    assert out["is_gap_filled"].sum() == 1
    assert (out["price_usd"] > 0).all()


def test_clean_coin_requires_raw_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess.clean_coin("bitcoin", tmp_path, PARAMS)


def test_flag_outliers_detects_price_spike():
    rng = np.random.default_rng(1)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.001, 150))))
    price.iloc[80] *= 1.5
    flags = preprocess.flag_outliers(price)
    assert bool(flags.iloc[80])
    assert flags.sum() <= 3
