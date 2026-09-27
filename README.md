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


## Sumber Video: File, Webcam, LAN / Online Stream

`main.py` mendukung beberapa tipe sumber video lewat argumen `--source`:

- File video lokal (`.mp4`, `.avi`, ...)
- Webcam / device number (numeric index, mis. `0`)
- RTSP (kamera LAN), HLS (`.m3u8`) atau HTTP stream
- Re-stream lokal (mis. menggunakan `ffmpeg`)

Contoh singkat:

- File video lokal:

```bash
python main.py --source "video_tes.mp4" --name "Tes-File"
```

- Webcam (device index):

```bash
python main.py --source 0 --name "Webcam-Depan"
```

- RTSP (kamera LAN / IP camera):

```powershell
python main.py --source "rtsp://user:pass@192.168.1.50:554/stream1" --name "Gate-LAN-1"
```

- HLS (.m3u8) atau HTTP stream:

```powershell
python main.py --source "http://192.168.1.50:8080/live/stream.m3u8" --name "Gate-HLS"
```

Prinsip & tips penting:

- Device numeric: `main.py` mengubah `--source` ke `int` jika seluruh string hanya angka (campur angka+huruf → dianggap URL/file).
- Jika stream RTSP gagal (frame kosong), coba put `rtsp_transport=tcp` melalui `ffmpeg` re-streamer atau gunakan `ffmpeg -rtsp_transport tcp -i "RTSP_URL" ...`.
- HLS (.m3u8) kadang memiliki segment latency; untuk real-time gunakan RTSP bila tersedia.

Menguji URL sebelum dipakai:

- VLC: `Media → Open Network Stream` — paling cepat untuk verifikasi.
- `ffprobe` untuk debug:

```bash
ffprobe "rtsp://user:pass@192.168.1.50:554/stream1"
```

Menggunakan `ffmpeg` sebagai proxy / re-streamer (rtsp -> local HTTP):

```bash
# contoh: jalankan re-stream lokal di port 8090
ffmpeg -rtsp_transport tcp -i "rtsp://user:pass@192.168.1.50:554/stream1" -f mpegts http://0.0.0.0:8090/feed1
```

Kemudian arahkan `main.py` ke feed lokal:

```powershell
python main.py --source "http://127.0.0.1:8090/feed1" --name "Gate-Proxy"
```

Menjaga kredensial aman (contoh env vars):

```powershell
#$env:CAM_URL = "rtsp://user:pass@192.168.1.50:554/stream1"
python main.py --source $env:CAM_URL --name "Gate-LAN-Env"
```

atau di bash:

```bash
export CAM_URL="rtsp://user:pass@192.168.1.50:554/stream1"
python main.py --source "$CAM_URL" --name "Gate-LAN-Env"
```

Troubleshooting umum:

- OpenCV / VideoCapture membaca frame kosong: periksa URL, network, dan coba VLC/ffmpeg. Coba re-stream dengan `-rtsp_transport tcp`.
- Authentication failed: pastikan username/password benar dan URL path sesuai vendor (ONVIF tools membantu menemukan URL).
- Latency / high CPU: turunkan resolusi stream (kamera side) atau tingkatkan `OCR_EVERY_N_FRAMES` di `main.py`.
- Banyak kamera: jalankan satu proses per 1-2 kamera atau gunakan NVR/aggregator.

Jika butuh, saya bisa menambahkan contoh konfigurasi `systemd`/Windows service untuk menjalankan `main.py` sebagai service per kamera.


## Kepatuhan Privasi Data & UU PDP

Sesuai UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi (UU PDP):


## Lisensi
Dikembangkan di bawah lisensi MIT. Bebas digunakan dan dikembangkan untuk otomatisasi lalu lintas dan gerbang keamanan di Indonesia.


Jika Anda ingin saya menambahkan contoh konfigurasi, sistem service (Windows service / systemd), atau
meningkatkan dokumentasi bahasa Inggris, beri tahu saya.
```
