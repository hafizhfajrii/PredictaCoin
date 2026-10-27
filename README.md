# PredictaCoin

Sistem Klasifikasi Tren dan Deteksi Anomali Cryptocurrency (BTC & ETH) dengan **Continual Learning**.

---

## 1. Tujuan Proyek

Proyek ini membangun sistem machine learning yang mengklasifikasikan arah tren harga
(Naik / Turun / Stabil) untuk Bitcoin (BTC) dan Ethereum (ETH) dalam horizon 1 jam ke
depan, sekaligus mendeteksi anomali pasar (crash mendadak, pump-and-dump).

Karena pasar cryptocurrency sangat volatil dan pola datanya terus berubah (data drift &
concept drift), sistem ini dirancang dengan pendekatan Continual Learning: model
secara otomatis mengambil data baru, dilatih ulang secara berkala, dan dipantau
performanya agar tetap relevan terhadap kondisi pasar terkini.

## 2. Struktur Direktori

Struktur proyek mengikuti konvensi *Cookiecutter Data Science*:

```
PredictaCoin/
├── .devcontainer/
│   └── devcontainer.json      # Konfigurasi GitHub Codespaces
├── .github/
│   └── workflows/
│       └── ingest.yml         # GitHub Actions: pipeline data tiap jam
├── config/
│   └── config.yaml            # Parameter proyek (aset, threshold, ingestion, preprocessing, dsb)
├── data/
│   ├── raw/
│   │   ├── {coin}/            # Data mentah per coin per bulan (YYYY-MM.parquet)
│   │   └── snapshots/         # Respons JSON mentah ber-timestamp tiap run
│   ├── interim/               # Data hasil preprocessing (per coin)
│   ├── processed/             # Fitur + label (features.parquet, latest.parquet, metadata.json)
│   └── external/              # Data pendukung dari sumber lain (jika ada)
├── models/                    # Model hasil training (versi terbaru & histori)
├── notebooks/                 # Notebook eksplorasi (EDA) dan eksperimen
├── src/
│   ├── data/
│   │   ├── ingest_data.py     # Data ingestion dari CoinGecko API
│   │   ├── preprocess.py      # Preprocessing / cleaning data mentah
│   │   └── run_pipeline.py    # Entry point: ingest -> preprocess -> features
│   ├── features/
│   │   └── build_features.py  # Feature engineering (indikator teknikal, label)
│   ├── models/                # Skrip training, evaluasi, dan inference
│   └── monitoring/            # Skrip pemantauan drift & performa model
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## 3. Persiapan Lingkungan

### 3.1. Via GitHub Codespaces (disarankan)

1. Buka repositori PredictaCoin di GitHub.
2. Klik tombol "<> Code" berwarna hijau di bagian atas halaman.
3. Pindah ke tab "Codespaces", lalu pilih "Create codespace on main" (atau pada branch yang diinginkan).
4. Tunggu hingga proses inisialisasi environment selesai (Python 3.11).
5. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

### 3.2. Lokal

```bash
git clone https://github.com/hafizhfajrii/PredictaCoin.git
cd PredictaCoin
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Gunakan Python 3.11 agar sama dengan runner GitHub Actions.

### 3.3. API key CoinGecko

Skrip ingestion membutuhkan **CoinGecko Demo API key** (gratis, daftar di coingecko.com).

1. Buat file bernama `.env` di root folder proyek (file ini sudah di-ignore oleh Git).
2. Isi dengan satu baris:
   ```
   COINGECKO_API_KEY=isi_api_key_kamu
   ```

Di GitHub Actions, key dibaca dari repository secret `COINGECKO_API_KEY`
(Settings → Secrets and variables → Actions).

## 4. Cara Menjalankan

Semua perintah dijalankan dari **root folder proyek**.

### 4.1. Pipeline lengkap (ingest → preprocess → features)

```bash
python src/data/run_pipeline.py
```

Pipeline berhenti dengan exit code 1 jika ada tahap yang gagal.

### 4.2. Tahap per tahap

```bash
# 1. Ingestion: ambil data dari CoinGecko ke data/raw/
python src/data/ingest_data.py

# 2. Preprocessing: bersihkan data/raw/ -> data/interim/
python src/data/preprocess.py

# 3. Feature engineering: data/interim/ -> data/processed/
python src/features/build_features.py
```

### 4.3. Opsi `ingest_data.py`

| Opsi | Fungsi | Contoh |
|------|--------|--------|
| *(tanpa opsi)* | Backfill 90 hari jika data belum ada, selanjutnya incremental 2 hari | `python src/data/ingest_data.py` |
| `--coins` | Override daftar coin | `--coins bitcoin` |
| `--days` | Override jumlah hari | `--days 7` |
| `--no-snapshot` | Tidak menyimpan snapshot JSON mentah | `--no-snapshot` |

### 4.4. Simulasi pengambilan data berkala

Skrip aman dijalankan berulang kali. Untuk mensimulasikan pengambilan periodik secara manual,
jalankan `python src/data/ingest_data.py` beberapa kali dengan jeda waktu. Setiap run akan:

- menambah satu snapshot baru di `data/raw/snapshots/` (file lama tidak ditimpa),
- menambah jam baru ke `data/raw/{coin}/{YYYY-MM}.parquet` tanpa menghapus data lama,
- tidak menghasilkan duplikat, meskipun rentang data tumpang tindih.

Secara otomatis, GitHub Actions (`.github/workflows/ingest.yml`) menjalankan pipeline **tiap jam**
(cron `0 * * * *`) dan bot meng-commit perubahan data ke branch `main`. Workflow juga bisa dijalankan
manual lewat tab Actions → *Data Pipeline (hourly)* → *Run workflow*.

## 5. Penjelasan Skrip

### 5.1. `src/data/ingest_data.py` (Data Ingestion)

**Sumber:** endpoint `/coins/{id}/market_chart` CoinGecko (`vs_currency=usd`; granularitas per jam
otomatis untuk rentang 2–90 hari). Data: harga, market cap, dan volume 24 jam.

**Pengambilan berkala tanpa menimpa data lama (non-destruktif):**

| Mekanisme | Penjelasan |
|-----------|------------|
| Backfill / incremental | Run pertama mengambil 90 hari (2.160 titik per coin); run berikutnya 2 hari terakhir dengan data tumpang tindih sehingga jam yang terlewat terisi otomatis |
| Append + dedup | Data baru digabung ke file bulanan, duplikat dibuang berdasarkan `(coin_id, timestamp)`, baris terbaru menang |
| Penulisan atomik | Parquet ditulis ke file sementara lalu di-replace, sehingga file lama tidak rusak jika proses terhenti |
| Snapshot ber-timestamp | Respons JSON mentah disimpan sebagai `data/raw/snapshots/{coin}_{YYYYMMDDTHHMMSSZ}.json` (retensi 7 hari, bisa diatur di `config.yaml`; data parquet tidak ikut dihapus) |

**Penanganan error koneksi dan respons:**

| Kondisi | Penanganan |
|---------|------------|
| Timeout / koneksi putus | Retry hingga 5 kali dengan exponential backoff (10, 20, 40, 80 detik) |
| HTTP 429 (rate limit) | Retry; header `Retry-After` dihormati jika lebih lama dari backoff |
| HTTP 5xx | Retry dengan backoff |
| Respons bukan JSON | Retry dengan backoff |
| HTTP 401 / 403 / 404 | Gagal langsung dengan pesan petunjuk (cek API key / coin_id), tanpa retry |
| Respons kosong atau kunci hilang | Ditolak sebelum disimpan (`IngestionError`) |
| Satu coin gagal | Coin lain tetap diproses; exit code 1 di akhir agar pipeline berhenti |

### 5.2. `src/data/preprocess.py` (Preprocessing)

Data proyek ini berupa **time series numerik**, sehingga langkah teks seperti tokenization dan
stopword removal tidak relevan. Preprocessing difokuskan pada kualitas deret waktu:

| No | Langkah | Alasan |
|----|---------|--------|
| 1 | Gabung seluruh partisi bulanan dan validasi kolom wajib | Memastikan input lengkap |
| 2 | Seragamkan timestamp (UTC, jam penuh) dan paksa kolom numerik | Konsistensi tipe data |
| 3 | Hapus duplikat `(coin_id, timestamp)`, data terbaru menang | Data tumpang tindih antar-run |
| 4 | Nilai tidak valid (inf, harga ≤ 0, market cap/volume negatif) dijadikan missing | Mencegah nilai rusak masuk ke fitur |
| 5 | Reindex ke deret jam lengkap; isi gap dengan forward-fill maksimal 3 jam, ditandai `is_gap_filled`; gap lebih panjang dibuang | Fitur rolling/lag butuh jarak antar-baris yang seragam |
| 6 | Flag outlier dengan robust z-score (median/MAD) pada log return 1 jam, jendela 7 hari, ambang 5,0 (`is_outlier`) | Outlier **tidak dibuang** karena crash/pump adalah sinyal yang ingin dideteksi |
| 7 | Validasi skema dan rentang dengan `pandera` (tipe, unik, harga > 0, dst.) | Menjamin data interim siap untuk ekstraksi fitur |

Parameter (`max_ffill_hours`, `outlier_window_hours`, `outlier_z_threshold`) dapat diubah di bagian
`preprocessing` pada `config/config.yaml`.

## 6. Contoh Output

Contoh data mentah (`data/raw/bitcoin/2026-10.parquet`), skema:

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| timestamp | datetime (UTC) | Dibulatkan ke jam penuh |
| coin_id | string | bitcoin / ethereum |
| price_usd | float | Harga (USD) |
| market_cap_usd | float | Kapitalisasi pasar (USD) |
| total_volume_usd | float | Volume 24 jam (USD) |
| fetched_at | datetime (UTC) | Waktu data diambil (audit) |

Cara melihat sampel data:

```bash
python -c "import pandas as pd; print(pd.read_parquet('data/raw/bitcoin/2026-10.parquet').tail())"
```

Contoh log run incremental (format ringkas):

```
2026-10-26 03:00:01 [INFO] [INCREMENTAL] bitcoin: days=2
2026-10-26 03:00:02 [INFO]   status OK, 48 titik, 2026-10-24 04:00:00+00:00 s/d 2026-10-26 03:00:00+00:00
2026-10-26 03:00:02 [INFO]   snapshot -> data/raw/snapshots/bitcoin_20261026T030001Z.json
2026-10-26 03:00:02 [INFO]   -> data/raw/bitcoin/2026-10.parquet (597 baris)
```

## 7. Pengecekan Kualitas Kode

Kode mengikuti PEP 8. Untuk mengecek:

```bash
pip install flake8
flake8 src/data --max-line-length=100
```

## 8. Pemecahan Masalah

| Gejala | Penyebab / solusi |
|--------|-------------------|
| `COINGECKO_API_KEY tidak ditemukan` | Buat `.env` di root proyek (lihat 3.3) |
| `HTTP 401/403 (cek COINGECKO_API_KEY)` | API key salah atau belum aktif |
| `[retry n/5] ... HTTP 429` | Rate limit; skrip menunggu otomatis. Jangan jalankan terlalu sering |
| `Tidak ada data raw untuk ...` saat preprocessing | Jalankan `ingest_data.py` terlebih dulu |
| `ModuleNotFoundError` | Jalankan `pip install -r requirements.txt` |

## 9. Strategi Branching (GitHub Flow)

Proyek ini menerapkan **GitHub Flow**:

1. Branch `main` selalu dalam kondisi stabil dan dapat dijalankan.
2. Setiap eksperimen atau fitur baru dikerjakan di branch terpisah (contoh: `feature/lk04-ingestion`).
3. Perubahan di-commit secara bertahap dengan pesan commit yang jelas.
4. Setelah selesai dan divalidasi, branch didorong ke GitHub dan dibuka **Pull Request** ke `main`.
5. Pull Request direview sebelum di-merge ke `main`.
6. Branch fitur dihapus setelah merge untuk menjaga kebersihan repositori.

Catatan: commit data otomatis (`data: ingest ...`) dibuat oleh bot langsung ke `main`, sehingga
sebelum membuat branch baru lakukan `git pull` (atau *Pull origin* di GitHub Desktop).

## 10. Lisensi

Proyek ini menggunakan lisensi **MIT**
