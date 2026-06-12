# MLOps_Predictive-Maintanance-NASA-C-MAPSS

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predictive Maintanance untuk Mesin Industri Berbasis Continual Learning Menggunakan NASA C-MAPSS

## Tujuan
- Memprediksi waktu kegagalan mesin sebelum benar-benar terjadi sehingga tindakan pemeliharaan dapat dilakukan secara proaktif
- Mengoptimalkan jadwal perawatan
- Mengurangi downtime tidak terencana
- Menekan biaya operasional

## Cara Menjalankan Codespaces
1. Buka repository Github https://github.com/kaneysha/MLOps_Predictive-Maintenance-NASA-C-MAPSS
2. Klik tombol 'Code'
3. Klik tab 'Codespaces'
4. Klik "Create codespace on main"
5. Tunggu hingga proses build selesai

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mlops_predictive_maintanance_nasa_c_mapss and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mlops_predictive_maintanance_nasa_c_mapss   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops_predictive_maintanance_nasa_c_mapss a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Data Ingestion

Tujuan: Mengambil data mentah statis dari file eksternal (train_FD001.txt) secara berkala dan menyimpannya ke folder data/raw/ dalam batch kecil untuk mendukung Continual Learning.

Jalankan: python src/ingest_data.py

Script ini akan:
- Membaca file eksternal TXT FD001
- Menghapus kolom kosong (dropna(axis=1)) dan memberi nama kolom sesuai sensor dan setting operasi
- Membuat folder data/raw/ jika belum ada
- Membagi data menjadi batch kecil (batch_size=50) agar bisa diproses bertahap
- Menyimpan setiap batch ke file CSV dengan timestamp unik supaya file lama tidak tertimpa
- Memberikan delay 5 detik antar batch untuk simulasi data streaming
- Mencetak status file yang disimpan

## Preprocessing

Tujuan: Skrip ini membersihkan data mentah dari folder data/raw/ dan menyimpan hasilnya ke data/processed/cleaned_data.csv agar data mentah siap digunakan untuk feature extraction dan training model tanpa duplikasi.

Jalankan: python src/preprocess.py

Script ini akan:
- Mencari semua file CSV di folder data/raw/
- Membaca log processed_log.txt untuk melacak file yang sudah diproses agar tidak diproses ulang
- Looping tiap file baru:
  - Print status
  - Membaca file CSV ke DataFrame
  - Mengisi nilai kosong (fillna(method='ffill'))
  - Menghapus kolom unit jika ada
  - Menyimpan DataFrame bersih ke list sementara
- Menggabungkan semua data baru menjadi satu CSV (cleaned_data.csv)
- Update processed_log.txt untuk menandai file yang sudah diproses
- Print status akhir

--------

## Versioning Data dengan DVC
1. Menambahkan dataset
   Dataset yang akan disimpan diletakkan di folder data terlebih dahulu
2. Instalasi dan inisialisasi DVC
   pip install dvc
   dvc init            -> membuat file konfigurasi .dvc/ dan .dvcignore
   git add .dvc .dvcignore
   git commit -m "Initialize DVC"    -> menyimpan konfigurasi ke Git
3. Tracking dataset dengan DVC
   dvc add data    -> membuat file data.dvc, menyimpan metadata veri data, dan menambahkan ke cache DVC
4. Menyimpan perubahan ke Git
   git add data.dvc .gitignore
   git commit -m "add data"    -> dataset tidak disimpan langsung ke Git, hanya metadata DVC yang dicommit
5. Konfigurasi remote storage DVC
   dvc remote add -d localstorage ../dvc-storage    -> remote storage digunakan untuk menyimpan file dataset yang sebenarnya.
   git add .dvc/config
   git commit -m "Configure DVC remote storage"
6. Mengunggah data ke remote storage DVC
   dvc push    -> mengambil data dari cache DVC lalu mengunggah ke remote storage
7. Membuat versi dataset baru
   dvc add data                            -> perbarui dataset
   git add data.dvc
   git commit -m "Update dataset version"  -> simpan perubahan
   dvc push                                -> kirim dataset baru ke remote storage
8. Melacak riwayat perubahan dataset
   dvc diff    -> membandingkan dua versi dataset yang berbeda berdasarkan riwayat Git
   dvc status  -> memeriksa apakah terdapat perubahan pada data atau pipeline

--------

## Model Versioning
Saat ini layanan inferensi menggunakan Version 1 dari model nasa_cmapss_model yang berada pada stage Production. Model ini dipilih karena telah ditetapkan sebagai model produksi yang stabil dan tervalidasi. Versi yang lebih baru, yaitu Version 9, masih berada pada stage Staging untuk proses pengujian dan validasi lebih lanjut sebelum dipromosikan ke Production.

--------

## Docker Compose
Seluruh komponen sistem dapat dijalankan secara bersamaan menggunakan Docker Compose.

1. Membangun dan menjalankan sistem
   docker compose up --build -d    -> membangun image, menajlankan seluruh container, membuat jaringan antarlayanan secara otomatis
2. Memastikan container berjalan
   docker ps                       -> melihat docker yang sedang aktif
3. Menghentikan sistem
   docker compose down

--------

## Mengakses endpoint API
Setelah seluruh container berhasil dijalankan, layanan API dapat diakses melalui http://localhost:8000
Endpoint utama untuk melakukan inferensi adalah: POST /predict

Contoh request menggunakan cURL:
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d @predict.json

Dokumentasi interaktif FastAPI dapat diakses melalui http://localhost:8000/docs
Dokumentasi OpenAPI tersedia pada http://localhost:8000/openapi.json

## Menambah Jumlah Replika Secara Dinamis
Contoh 3 replika service API:
docker compose up --scale api-service=3 -d

Nginx Load Balancer akan mendistribusikan request secara otomatis ke seluruh replika yang aktif sehingga beban inferensi dapat dibagi secara merata.

--------

## Konfigurasi Ambang Batas Alert
1. Alert Penurunan Akurasi Model
Metrik yang digunakan: model_accuracy
Kondisi alert: model_accuracy < 0.8

Jika nilai akurasi turun di bawah 0,8, Grafana akan mengubah status alert menjadi Firing dan mengirimkan notifikasi. Kondisi ini mengindikasikan bahwa performa model telah menurun secara signifikan dibandingkan performa awal saat deployment.

2. Alert Data Drift
Metrik yang digunakan: data_drift_score
Kondisi alert: data_drift_score > 0.5

Jika nilai data drift score melebihi 0,5, Grafana akan mengaktifkan alert karena dianggap terdapat perbedaan distribusi data yang cukup signifikan dibandingkan data yang digunakan saat pelatihan model.
