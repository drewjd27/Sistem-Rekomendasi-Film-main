# Sistem Rekomendasi Film: Content-Based & Collaborative Filtering

Proyek ini membangun sistem rekomendasi film menggunakan dua pendekatan utama: Content-Based Filtering dan Collaborative Filtering. Sistem ini bertujuan membantu pengguna menemukan film yang relevan sesuai preferensi mereka, serta membandingkan keunggulan dan kekurangan kedua pendekatan.

## Dataset

- **Sumber:** [Kaggle - The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- **File utama:** `movies_metadata.csv`, `ratings_small.csv`

## Fitur Utama

- **Content-Based Filtering:**  
  Rekomendasi berdasarkan kemiripan konten (genre) antar film menggunakan TF-IDF dan cosine similarity.
- **Collaborative Filtering:**  
  Rekomendasi berdasarkan pola interaksi pengguna menggunakan model deep learning (embedding).

## Struktur Project

- `notebook.ipynb` — Notebook utama berisi seluruh pipeline, EDA, modeling, dan evaluasi.
- `README.md` — Dokumentasi singkat project.
- `laporan.md` — Laporan dari proyek sistem rekomendasi ini.
- Dataset berada di folder `dataset/`. Pastikan setelah kamu mengunduh dataset, kamu menaruhnya di folder `dataset/` agar notebook dapat membaca dataset dengan benar.
- `notebook.py` — Script yang dapat dijalankan secara langsung menggunakan `python notebook.py` di terminal.
- `requirements.txt` — Daftar library yang digunakan dalam proyek ini.

## Cara Menjalankan

1. Pastikan seluruh dependensi (lihat di notebook) telah terinstal.
2. Jalankan notebook `notebook.ipynb` secara berurutan.
3. Ikuti penjelasan dan visualisasi pada setiap sel untuk memahami proses dan hasil.

## Hasil & Evaluasi

- **Content-Based Filtering:**  
  Memberikan rekomendasi film dengan genre serupa, precision dan recall tinggi untuk genre yang spesifik.
- **Collaborative Filtering:**  
  Memberikan rekomendasi personal berdasarkan pola rating pengguna lain, dievaluasi menggunakan MAE dan RMSE.