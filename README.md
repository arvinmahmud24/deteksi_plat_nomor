# High-Speed Indonesian ANPR & Gate Logger System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-blueviolet?style=for-the-badge)](https://github.com/ifzhang/ByteTrack)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-Async_Worker-FF6F00?style=for-the-badge)](https://github.com/JaidedAI/EasyOCR)
[![OpenPyXL](https://img.shields.io/badge/OpenPyXL-Excel_Embed-217346?style=for-the-badge)](https://openpyxl.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

Sistem Pengenal Plat Nomor Otomatis (ANPR / ALPR) khusus wilayah Indonesia yang mengintegrasikan deteksi objek **YOLOv8**, pelacakan **ByteTrack**, pembacaan karakter asinkron **Async Worker Queue EasyOCR**, serta pencatatan log harian otomatis ke spreadsheet Excel (`.xlsx`) lengkap dengan **foto crop plat nomor ter-embed presisi di Kolom F**.


## Ringkasan Proyek

Sistem ini dirancang untuk memantau stream CCTV (RTSP/HTTP Live Stream `.m3u8`/Webcam/File Video) secara real-time 24/7. Sistem memvalidasi format plat nomor Indonesia (awalan 1-2 huruf, 1-4 angka, 0-3 huruf akhiran), menerapkan *Multi-Frame Weighted Confidence Voting* per kendaraan, dan menyimpan foto bukti berwarna asli maupun versi *Grayscale/CLAHE preprocessed*.


## Fitur Utama

  Inferensi OCR dijalankan secara paralel pada *background thread worker pool*, menghasilkan GUI preview video yang *zero-lag* (FPS tetap tinggi).
  Mengakumulasi hasil OCR dari beberapa frame per ID kendaraan (`ByteTrack`). Hasil dengan confidence tertinggi dikunci saat kendaraan melintas.
  Setiap hari sistem otomatis membuat/memperbarui file Excel log harian tersendiri, membuat pengarsipan dan auditing data gerbang menjadi sangat rapi.
  Foto crop plat nomor Grayscale / Preprocessed di-embed secara otomatis dan presisi ke dalam sel Excel di Kolom F dengan ukuran teratur.
  Menyimpan foto warna asli di `storage/snapshots/` dan foto Grayscale/CLAHE di `storage/snapshots_grayscale/`.
  Dilengkapi script `prepare_snapshot_dataset.py` yang otomatis memotong karakter dari snapshot CCTV menjadi dataset karakter baru untuk melatih ulang model PyTorch CNN (`train_character.py`).
  Mencetak link URL lokal di terminal yang dapat diklik langsung (`Ctrl + Click`) untuk membuka foto bukti tanpa membuka Windows Explorer manual.
  Badge overlay pada preview video menggunakan font khusus plat nomor Indonesia (`PlatNomor-WyVnn.ttf`).
  Jika file Excel sedang dibuka di Microsoft Excel, data otomatis diselamatkan ke `data/log_plat_backup_YYYY-MM-DD.csv`.


## Struktur Direktori Proyek

```text
d:\PLAT\
├── main.py                   # Script aplikasi ANPR utama
├── prepare_snapshot_dataset.py # Auto generator dataset karakter dari foto snapshot CCTV
├── train_character.py        # Script pelatihan CNN untuk dataset karakter plat nomor
├── README.md                 # Dokumentasi proyek & panduan penggunaan
│
├── models/                   # Bobot Model Machine Learning
│   ├── plate.pt              # Model YOLOv8 deteksi plat nomor Indonesia
│   ├── char_model.pth        # Model PyTorch terlatih klasifikasi 36 karakter
│   └── char_labels.json      # Pemetaan indeks kelas karakter
│
├── assets/                   # Aset Statis Proyek
│   └── fonts/                # Typografi resmi plat nomor Indonesia
│       ├── PlatNomor-WyVnn.ttf
│       └── PlatNomor-eZ2dm.otf
│
├── data/                     # Output Log Excel Harian & Database
│   ├── log_plat_YYYY-MM-DD.xlsx # Log Excel harian dengan embedded crop plat
│   ├── log_plat_backup_YYYY-MM-DD.csv # Backup CSV harian (saat Excel dikunci)
│   └── gate_log.db           # Database SQLite
│
├── storage/                  # Foto Bukti Snapshot
│   ├── snapshots/            # Snapshot foto crop warna asli (BGR)
│   └── snapshots_grayscale/  # Snapshot foto crop Grayscale / CLAHE
│
└── dataset/                  # Dataset Pelatihan Karakter
    └── archive/              # Dataset Karakter (36 kelas: 0-9 & A-Z)
```


## Arsitektur Sistem

```text
   [ CCTV STREAM ] ──► main.py (YOLOv8 + ByteTrack + Async EasyOCR)
                             │
                             ├──► data/log_plat_YYYY-MM-DD.xlsx (Log & Embedded Crop)
                             │
                             └──► storage/snapshots/ (Foto Crop Plat Real-World)
                                       │
                                       ▼
                         prepare_snapshot_dataset.py (Auto Character Segmenter)
                                       │
                                       ▼
                         dataset/archive/DatasetCharacter/ (Dataset Karakter)
                                       │
                                       ▼
                         train_character.py (PyTorch CNN Training)
                                       │
                                       ▼
                         models/char_model.pth (Model Terlatih)
```


## Cara Menjalankan Aplikasi

### 1. Prasyarat & Instalasi Dependensi
Pastikan Python 3.10+ telah terinstall:
```bash
pip install ultralytics easyocr openpyxl opencv-python pillow numpy torch torchvision
```

### 2. Menjalankan ANPR
Untuk memulai pengawasan CCTV dan pencatatan log harian otomatis (menggunakan URL bawaan):
```bash
python main.py
```

Anda juga bisa menentukan sumber video, IP Camera (RTSP/HTTP), atau webcam secara dinamis melalui argumen command line:
```bash
# Menggunakan Webcam
python main.py --source 0 --name "Webcam Lokal"

# Menggunakan file video
python main.py --source "video_tes.mp4" --name "Video Testing"

# Menggunakan IP Camera / RTSP
python main.py --source "rtsp://admin:123@192.168.1.10:554/stream" --name "Gate 1"
```

### 3. Ekstraksi Dataset Karakter dari Snapshot CCTV
Untuk memotong snapshot CCTV menjadi sampel dataset karakter baru:
```bash
python prepare_snapshot_dataset.py
```

### 4. Melatih Ulang Model Karakter (Opsional)
Untuk melatih ulang model Neural Network karakter PyTorch:
```bash
python train_character.py
```


## Menggunakan Kamera LAN (RTSP / HLS)

Jika kamera Anda ada di LAN, gunakan URL RTSP atau HLS langsung sebagai `--source`. Langkah singkat:

- Pastikan kamera dan mesin yang menjalankan `main.py` berada pada jaringan yang sama (subnet) atau ada routing.
- Berikan IP statis atau DHCP reservation ke kamera (mis. `192.168.1.50`).
- Aktifkan RTSP/ONVIF pada pengaturan kamera dan catat username/password.

Contoh URL umum:

- RTSP:
  - `rtsp://user:pass@192.168.1.50:554/stream1`
  - `rtsp://user:pass@192.168.1.50:554/h264`
- HLS (.m3u8):
  - `http://192.168.1.50:8080/live/stream.m3u8`

Contoh menjalankan `main.py` dengan sumber LAN (PowerShell):

```powershell
python main.py --source "rtsp://user:pass@192.168.1.50:554/stream1" --name "Gate-LAN-1"
```

Jika VideoCapture/FFmpeg gagal karena transport (UDP vs TCP), Anda dapat re-stream RTSP menjadi HTTP lokal menggunakan `ffmpeg` dan memakai URL lokal di `main.py`.

Contoh re-streamer (lihat `scripts/`):

- Linux / macOS (bash): `scripts/restream_rtsp.sh`
- Windows (PowerShell): `scripts/restream_rtsp.ps1`

Contoh perintah `ffmpeg` (rtsp -> local HTTP MPEG-TS):

```bash
ffmpeg -rtsp_transport tcp -i "rtsp://user:pass@192.168.1.50:554/stream1" -f mpegts http://0.0.0.0:8090/feed1
```

Lalu gunakan lokal HTTP URL di `main.py`:

```powershell
python main.py --source "http://127.0.0.1:8090/feed1" --name "Gate-LAN-1"
```

Keamanan & catatan:

- Jangan commit kredensial kamera ke repo. Simpan di environment variables atau manager secrets.
- Untuk production, gunakan RTSP proxy atau RTSP server (mis. RTSP Simple Server) untuk stabilitas.
- Jika Anda menggunakan banyak kamera, pertimbangkan NVR/aggregator atau jalankan worker terpisah per kamera.


## Kepatuhan Privasi Data & UU PDP

Sesuai UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi (UU PDP):


## Lisensi
Dikembangkan di bawah lisensi MIT. Bebas digunakan dan dikembangkan untuk otomatisasi lalu lintas dan gerbang keamanan di Indonesia.


Jika Anda ingin saya menambahkan contoh konfigurasi, sistem service (Windows service / systemd), atau
meningkatkan dokumentasi bahasa Inggris, beri tahu saya.
```
