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
│   └── config.yaml            # Parameter proyek (threshold label, jadwal retrain, dsb)
├── data/
│   ├── raw/                   # Data mentah hasil fetch dari CoinGecko API
│   │   └── snapshots/         # Respons JSON mentah ber-timestamp tiap run
│   ├── interim/               # Data hasil preprocessing (per coin)
│   ├── processed/             # Data yang sudah dibersihkan & di-label
│   └── external/              # Data pendukung dari sumber lain (jika ada)
├── models/                    # Model hasil training (versi terbaru & histori)
├── notebooks/                 # Notebook eksplorasi (EDA) dan eksperimen
├── src/
│   ├── data/                  # Skrip ingestion & preprocessing data
│   │   ├── ingest_data.py     # Pengumpul data dari CoinGecko API
│   │   ├── preprocess.py      # Pembersihan data mentah
│   │   └── run_pipeline.py    # Menjalankan ingest -> preprocess -> features
│   ├── features/              # Skrip feature engineering (indikator teknikal, dsb)
│   ├── models/                # Skrip training, evaluasi, dan inference
│   └── monitoring/            # Skrip pemantauan drift & performa model
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## 3. Cara Menjalankan via GitHub Codespaces

1. Buka repositori PredictaCoin di GitHub.
2. Klik tombol "<> Code" berwarna hijau di bagian atas halaman.
3. Pindah ke tab "Codespaces", lalu pilih "Create codespace on main".
4. Tunggu hingga proses inisialisasi environment selesai.
5. Buat file baru bernama .env di root folder proyek untuk menyimpan API key.
6. Tambahkan baris berikut ke dalam file .env: COINGECKO_API_KEY=<api_key>.
7. Jalankan python src/data/ingest_data.py di terminal untuk memastikan koneksi ke CoinGecko API berjalan dan data BTC/ETH berhasil ditarik.

## 4. Strategi Branching (GitHub Flow)

Proyek ini menerapkan **GitHub Flow**:

1. Branch `main` selalu dalam kondisi stabil dan dapat dijalankan.
2. Setiap eksperimen atau fitur baru dikerjakan di branch terpisah
3. Perubahan di-commit secara bertahap dengan pesan commit yang jelas.
4. Setelah selesai dan divalidasi, branch didorong ke GitHub dan dibuka **Pull Request** ke `main`.
5. Pull Request direview sebelum di-merge ke `main`.
6. Branch fitur dihapus setelah merge untuk menjaga kebersihan repositori.

## 5. LK-04: Pengumpulan Data dan Prapemrosesan

Bagian ini menjelaskan cara menjalankan skrip pengumpul data dan prapemrosesan.

### 5.1. Persiapan

1. Gunakan GitHub Codespaces (Python 3.11, lihat bagian 3) atau Python 3.11 lokal.
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Pastikan file `.env` berisi `COINGECKO_API_KEY=<api_key>` (CoinGecko Demo API key).

### 5.2. Cara Menjalankan

Jalankan dari root proyek.

```bash
# Pengumpulan data -> data/raw/
python src/data/ingest_data.py

# Prapemrosesan -> data/interim/
python src/data/preprocess.py

# Semua tahap sekaligus (ingest -> preprocess -> features)
python src/data/run_pipeline.py
```

Opsi tambahan `ingest_data.py`:

| Opsi | Fungsi |
|------|--------|
| `--coins bitcoin` | Hanya ambil coin tertentu |
| `--days 7` | Ambil jumlah hari tertentu |
| `--no-snapshot` | Tidak menyimpan snapshot JSON |

### 5.3. Cara Kerja Pengumpulan Data (`ingest_data.py`)

- **Sumber:** CoinGecko API `/coins/{id}/market_chart` (harga, market cap, volume; per jam).
- **Berkala:** run pertama mengambil 90 hari (backfill), run berikutnya 2 hari terakhir (incremental).
  Skrip dijalankan otomatis tiap jam oleh GitHub Actions (`.github/workflows/ingest.yml`).
- **Tidak menimpa data lama:** data baru ditambahkan ke file bulanan dan duplikat dibuang berdasarkan
  `(coin_id, timestamp)`. Setiap run juga menyimpan snapshot JSON mentah dengan timestamp pada nama file
  (`data/raw/snapshots/{coin}_{YYYYMMDDTHHMMSSZ}.json`).
- **Penanganan error:** timeout, koneksi putus, HTTP 429/5xx, dan respons bukan JSON di-retry hingga 5 kali
  dengan exponential backoff. HTTP 401/403/404 gagal langsung dengan pesan jelas. Respons kosong ditolak.

### 5.4. Prapemrosesan (`preprocess.py`)

Data berupa time series numerik, sehingga langkahnya berfokus pada missing values dan outlier:

1. Gabungkan semua file bulanan dan seragamkan timestamp (UTC, jam penuh).
2. Hapus duplikat `(coin_id, timestamp)`.
3. Nilai tidak valid (inf, harga <= 0, market cap/volume negatif) dijadikan missing.
4. Reindex ke deret jam lengkap, isi gap dengan forward-fill maksimal 3 jam (ditandai `is_gap_filled`).
5. Tandai outlier dengan robust z-score (kolom `is_outlier`), tidak dibuang karena crash/pump adalah sinyal.
6. Validasi skema dan rentang nilai dengan `pandera`.

### 5.5. Contoh Output

Log pengumpulan data (dipotong):

```
2026-10-27 06:23:42 [INFO] [INCREMENTAL] bitcoin: days=2
2026-10-27 06:23:42 [INFO]   status OK, 48 titik, 2026-10-25 07:00:00+00:00 s/d 2026-10-27 06:00:00+00:00
2026-10-27 06:23:42 [INFO]   snapshot -> data/raw/snapshots/bitcoin_20261027T062342Z.json
2026-10-27 06:23:42 [INFO]   -> data/raw/bitcoin/2026-10.parquet (631 baris)
2026-10-27 06:23:45 [INFO] bitcoin: raw=2510 -> dedup=2510 -> final=2510 | tidak valid=0, gap terisi=0, gap tak terisi=0, outlier=41
```

Pengujian dua kali run berurutan: jumlah baris `data/raw/bitcoin/2026-10.parquet` tetap 631 dengan 0 duplikat,
dan snapshot bertambah tanpa menghapus yang lama.

### 5.6. Pemecahan Masalah

| Pesan | Solusi |
|-------|--------|
| `COINGECKO_API_KEY tidak ditemukan` | Buat file `.env` (lihat bagian 3 dan 5.1) |
| `HTTP 401/403` | API key salah |
| `[retry n/5] ... HTTP 429` | Rate limit, skrip menunggu otomatis |
| `Tidak ada data raw untuk ...` | Jalankan `ingest_data.py` dulu |

## 6. LK-05: Data Versioning dengan DVC

**Mengapa data perlu diversioning.** Pada Continual Learning, data bertambah tiap jam dan model dilatih
ulang berkala. Tanpa catatan versi data, training tidak bisa diulang dan dua model tidak bisa dibandingkan
secara adil. Dengan DVC, setiap model dapat mencatat versi datanya (tag `data-vX.Y.Z`) sehingga dataset
yang persis sama dapat dikembalikan kapan saja (reproduksibilitas), dan setiap perubahan data tercatat
dengan hash (audit). Git hanya menyimpan file penunjuk `.dvc` (sekitar 100 byte), bukan data besarnya.

**Data yang dilacak.** `data/raw/` dikelola pipeline GitHub Actions (commit tiap jam), jadi tidak dilacak
langsung. `export_dataset.py` menggabungkan data mentah BTC dan ETH (hasil LK-04) menjadi
`data/dvc/dataset_raw.parquet` yang dilacak DVC.

**Persiapan (sekali saja).**

```bash
pip install "dvc>=3.0"
dvc init
dvc remote add -d penyimpanan $HOME/dvc-storage
```

**Alur menambah versi data baru.**

```bash
python src/data/ingest_data.py          # ambil data terbaru
python src/data/export_dataset.py       # ekspor dataset gabungan
dvc status                              # perubahan terdeteksi (modified)
dvc add data/dvc/dataset_raw.parquet    # hash di file .dvc berubah
git add data/dvc/dataset_raw.parquet.dvc
git commit -m "data: dataset versi baru"
git tag data-v1.0.1                     # PATCH: data bertambah, skema tetap
dvc push                                # kirim data ke remote
```

**Hasil simulasi dua versi.**

| | Versi 1 (`data-v1.0.0`) | Versi 2 (`data-v1.0.1`) |
|---|---|---|
| Hash `md5` | `e90f7ae6...` | `3381a186...` |
| Total baris | 5.086 | 5.090 |

```bash
dvc diff data-v1.0.0 data-v1.0.1 --show-hash
# Modified:
#     e90f7ae6..3381a186  data/dvc/dataset_raw.parquet
```

Kembali ke versi lama: `git checkout data-v1.0.0 -- data/dvc/dataset_raw.parquet.dvc` lalu `dvc checkout`.

**Catatan.** Remote saat ini folder lokal di luar repositori. Untuk storage eksternal (S3, GCP, MinIO),
cukup ganti URL remote dengan `dvc remote modify`; alur di atas tidak berubah.

## 7. Lisensi

Proyek ini menggunakan lisensi **MIT**
