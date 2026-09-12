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
├── config/
│   └── config.yaml            # Parameter proyek (threshold label, jadwal retrain, dsb)
├── data/
│   ├── raw/                   # Data mentah hasil fetch dari CoinGecko API
│   ├── processed/             # Data yang sudah dibersihkan & di-label
│   └── external/              # Data pendukung dari sumber lain (jika ada)
├── models/                    # Model hasil training (versi terbaru & histori)
├── notebooks/                 # Notebook eksplorasi (EDA) dan eksperimen
├── src/
│   ├── data/                  # Skrip ingestion & preprocessing data
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
7. Jalankan python src/data/fetch_coingecko.py di terminal untuk memastikan koneksi ke CoinGecko API berjalan dan data BTC/ETH berhasil ditarik.

## 4. Strategi Branching (GitHub Flow)

Proyek ini menerapkan **GitHub Flow**:

1. Branch `main` selalu dalam kondisi stabil dan dapat dijalankan.
2. Setiap eksperimen atau fitur baru dikerjakan di branch terpisah
3. Perubahan di-commit secara bertahap dengan pesan commit yang jelas.
4. Setelah selesai dan divalidasi, branch didorong ke GitHub dan dibuka **Pull Request** ke `main`.
5. Pull Request direview sebelum di-merge ke `main`.
6. Branch fitur dihapus setelah merge untuk menjaga kebersihan repositori.

## 5. Lisensi

Proyek ini menggunakan lisensi **MIT**