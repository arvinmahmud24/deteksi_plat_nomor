# High-Speed Indonesian ANPR & Gate Logger System (PLAT)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-blueviolet?style=for-the-badge)](https://github.com/ifzhang/ByteTrack)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-Async_Worker-FF6F00?style=for-the-badge)](https://github.com/JaidedAI/EasyOCR)
[![OpenPyXL](https://img.shields.io/badge/OpenPyXL-Excel_Embed-217346?style=for-the-badge)](https://openpyxl.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

Sistem Pengenal Plat Nomor Otomatis (ANPR / ALPR) khusus wilayah Indonesia yang mengintegrasikan deteksi objek **YOLOv8**, pelacakan **ByteTrack**, pembacaan karakter asinkron **Async Worker Queue EasyOCR & PyTorch CNN**, serta pencatatan log harian otomatis ke spreadsheet Excel (`.xlsx`) lengkap dengan **foto crop plat nomor ter-embed presisi di Kolom F**.

---

## Cara Kerja Sistem (System Architecture & Pipeline)

Sistem ini bekerja secara otomatis dan real-time melalui 7 tahap utama:

```text
[ Input Stream Video / Kamera ]
              │
              ▼
    1. Input Frame Acquisition (Webcam / DroidCam / RTSP / Video File)
              │
              ▼
    2. Deteksi Bounding Box & Auto-Merge (YOLOv8)
              │
              ▼
    3. Multi-Object Tracking (ByteTrack ID)
              │
              ▼
    4. Async Worker Queue & Preprocessing (Multithreading Pool)
              │
              ▼
    5. Hybrid OCR Engine (EasyOCR + PyTorch CNN Character Recognition)
              │
              ▼
    6. Pembersihan & Normalisasi Format Indonesia (RegEx & Confusable Mapping)
              │
              ▼
    7. Finalisasi Log & Embedding Excel (.xlsx & CSV Backup)
```

---

### Detail Penjelasan 7 Tahap Pipeline:

#### 1. Input Frame Acquisition (Stream Video & Auto-Reconnect)
- System menerima masukan video dari berbagai sumber seperti Webcam Local (`0`, `1`), DroidCam (WiFi HTTP/USB DirectShow), maupun IP Camera Stream (RTSP/HLS `.m3u8`).
- Dilengkapi mekanisme **Auto-Reconnect & Fallback URL**: Jika koneksi jaringan/kamera terputus sementara, sistem akan otomatis melakukan percobaan sambung ulang tanpa mengalihkan sumber kamera secara sepihak.

#### 2. Deteksi Objek Plat Nomor & Box Merging (YOLOv8)
- Model `YOLOv8` (`models/plate.pt`) memindai setiap frame untuk menemukan lokasi objek plat nomor kendaraan.
- **Logic Box Merging**: Jika plat terdeteksi secara terpisah (misal baris atas huruf awalan dan baris bawah angka terpotong menjadi 2 box oleh YOLO), sistem akan secara cerdas mengabungkan (*merge*) kotak-kotak tersebut menjadi 1 Bounding Box plat nomor yang utuh.

#### 3. Pelacakan Objek Kendaraan (ByteTrack Multi-Object Tracking)
- Setiap plat nomor yang terdeteksi diberi ID pelacakan unik (`Track ID`) menggunakan algoritma **ByteTrack**.
- Dengan pelacakan ini, kendaraan yang melintas dalam puluhan frame video akan diidentifikasi sebagai 1 objek kendaraan yang sama, mencegah terjadinya pembacaan berulang (spamming log).

#### 4. Async Worker Queue & Pra-pemrosesan (Multithreading OCR)
- Untuk menjaga frame rate GUI tetap tinggi (tanpa *lag* atau patah-patah), proses pembacaan teks OCR tidak dijalankan di thread utama visual, melainkan dimasukkan ke dalam **`AsyncOCRWorker`** berbasis `Queue` Multithreading.
- **Preprocessing Crop**: Bagian bawah plat (seperti tanggal/bulan pajak) dibuang 20%, kemudian citra di-upscale berwarna secara presisi agar karakter plat terbaca jelas oleh engine OCR.

#### 5. Hybrid OCR Engine (EasyOCR + PyTorch CNN)
- Sistem mendukung mode dual-engine:
  - **EasyOCR Engine**: Membaca susunan teks plat nomor berwarna secara natural.
  - **PyTorch CNN Character Recognition (`PlateCharNetV2`)**: Melakukan segmentasi vertical projection dan memprediksi karakter per karakter menggunakan arsitektur Residual Network.
- Kedua engine berkolaborasi, dan hasil dengan nilai kepastian (*confidence score*) tertinggi akan dipilih.

#### 6. Pembersihan & Normalisasi Plat Nomor Indonesia (`clean_plate`)
- Teks mentah dari OCR divalidasi dan disesuaikan dengan aturan format Plat Nomor Indonesia:
  - **Awalan**: 1–2 Huruf Kode Wilayah valid di Indonesia (contoh: `B`, `AB`, `DK`, `N`, `L`, dll).
  - **Angka**: 1–4 Digit Angka Registrasi.
  - **Akhiran**: 0–3 Huruf Seri Opsional.
- Menerapkan **Confusable Mapping**: Mengoreksi kesalahan pembacaan OCR yang tertukar antara huruf dan angka (seperti `'O'`/`'D'` $\leftrightarrow$ `'0'`, `'I'`/`'L'` $\leftrightarrow$ `'1'`, `'S'` $\leftrightarrow$ `'5'`, `'B'` $\leftrightarrow$ `'8'`).

#### 7. Multi-Frame Voting, Snapshot & Auto-Embed Excel
- **Weighted Voting**: Selama kendaraan berada di dalam jangkauan kamera, sistem mengumpulkan beberapa kandidat bacaan OCR. Saat kendaraan selesai melintas (keluar frame), sistem memilih hasil dengan kombinasi *confidence* & kualitas gambar terbaik.
- **Snapshot Storage**: Menyimpan 2 file gambar bukti ke dalam disk:
  1. `storage/snapshots/`: Gambar crop plat berwarna asli.
  2. `storage/snapshots_grayscale/`: Gambar crop versi grayscale.
- **Auto-Embed Excel**: Log harian secara otomatis ditulis ke file `data/log_plat_YYYY-MM-DD.xlsx`. Foto crop grayscale ditempelkan (*embedded*) langsung ke **Kolom F** di dalam tabel sel Excel. Jika file Excel sedang dibuka di aplikasi PC, log otomatis diselamatkan ke file backup `data/log_plat_backup_YYYY-MM-DD.csv`.

---

## Hubungan Sistem dengan Dataset & Siklus Pelatihan (Dataset Lifecycle)

Sistem ini memiliki **Siklus Umpan Balik Mandiri (Self-Learning Loop)** di mana hasil tangkapan kamera (*snapshots*) di dunia nyata otomatis dikonversi menjadi dataset baru untuk meningkatkan akurasi model PyTorch CNN dari waktu ke waktu.

```text
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                            LIVE ANPR (main.py)                          │
 │  Input Kamera ──> YOLOv8 Deteksi ──> OCR Baca Plat ──> Simpan Snapshots │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │  File Crop (.jpg)
                                      ▼  di storage/snapshots/
 ┌─────────────────────────────────────────────────────────────────────────┐
 │               DATASET GENERATOR (prepare_snapshot_dataset.py)            │
 │  • Ekstrak Teks Plat dari Nama File                                     │
 │  • Multi-Method Character Segmentation (Projection + CCA + Grid)        │
 │  • Data Augmentations (Rotasi, Noise, Blur, Warp, Morph)                │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │  Karakter terklasifikasi (32x32)
                                      ▼  di dataset/archive/DatasetCharacter/
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   MODEL TRAINER (train_character.py)                    │
 │  • Melatih PyTorch CNN (PlateCharNetV2 dengan Residual Blocks)          │
 │  • Weighted Sampling & Label Smoothing                                  │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │  Bobot Model Baru (.pth)
                                      ▼  di models/char_model.pth
 ┌─────────────────────────────────────────────────────────────────────────┐
 │              AKURASI MENINGKAT PADA DETEKSI SANGAT PRESISI               │
 └─────────────────────────────────────────────────────────────────────────┘
```

### Detail Kaitan & Siklus Komponen Dataset:

#### A. Pengumpulan Data Otomatis (`storage/snapshots/`)
- Saat `main.py` beroperasi, setiap kali plat nomor berhasil dibaca dengan *confidence score* yang baik, sistem akan menyimpan foto potongannya ke `storage/snapshots/crop_YYYYMMDD_HHMMSS_PLATTEXT.jpg`.
- Nama file ini menyimpan **Ground Truth Teks Plat Nomor** (contoh: `crop_20260927_175139_AB1287KP.jpg` yang berarti plat tersebut bertuliskan `AB 1287 KP`).

#### B. Ekstraksi & Pemotongan Karakter (`prepare_snapshot_dataset.py`)
- Script ini bertugas mengubah foto plat utuh menjadi dataset per-karakter (`0-9`, `A-Z`):
  1. **Parsing Ground Truth**: Membaca teks `AB1287KP` dari nama file.
  2. **Multi-Method Character Segmentation**:
     - *Method 1 (Vertical Projection Profiling)*: Memotong karakter berdasarkan grafik piksel vertikal (gap antar huruf).
     - *Method 2 (Connected Component Analysis / CCA)*: Fallback jika huruf saling menempel.
     - *Method 3 (Proportional Grid Slicing)*: Fallback pembagi rata sesuai jumlah karakter.
  3. **Resizing**: Mengubah setiap potong karakter menjadi ukuran standar $32 \times 32$ piksel.
  4. **Data Augmentation**: Membuat 5 variasi buatan per karakter (rotasi $\pm 5^\circ$, kecerahan/kontras, blur, noise, serta distortif morfologi) untuk memperkaya dataset.
  5. **Directing Output**: Gambar disimpan otomatis ke direktori kelasnya masing-masing di `dataset/archive/DatasetCharacter/<KARAKTER>/`.

#### C. Pelatihan Model Klasifikasi Karakter (`train_character.py`)
- Script ini menggunakan dataset dari `DatasetCharacter/` untuk melatih arsitektur **`PlateCharNetV2`** (Deep CNN dengan Residual Connections):
  - **Balancing Class Imbalance**: Menerapkan *Weighted Random Sampler* agar karakter yang jarang muncul (seperti huruf `Z` atau angka `9`) mendapatkan perhatian seimbang dalam training.
  - **Evaluasi Per-Kelas & Early Stopping**: Mencegah *overfitting* dan menghasilkan file model `models/char_model.pth` serta mapping `models/char_labels.json`.

#### D. Pemanfaatan Kembali oleh System ANPR Utama (`main.py`)
- Saat `main.py` dijalankan kembali, fungsi `load_cnn_ocr_model()` akan membaca `models/char_model.pth`.
- Jika model ini ada, `main.py` beralih ke **Mode Hybrid (EasyOCR + PyTorch CNN)**, di mana pembacaan karakter menjadi jauh lebih presisi dan tahan terhadap gangguan pencahayaan maupun plat kotor.

---

## 🌟 Fitur Utama
=======
## Fitur Utama
>>>>>>> 4d72cdca27bc0d54850ae4f5af34ab2133d517a5

- **Async Worker Queue (`AsyncOCRWorker`)**: Inferensi OCR berjalan di background thread pool sehingga preview video tetap lancar tanpa *lag*.
- **ByteTrack & Multi-Frame Voting**: Mengakumulasi hasil OCR dari beberapa frame per ID kendaraan untuk mengunci hasil dengan confidence tertinggi.
- **Log Excel Harian Auto-Embed**: Log tersimpan di `data/log_plat_YYYY-MM-DD.xlsx` dengan foto crop grayscale plat nomor langsung ter-embed di Kolom F.
- **Backup CSV Otomatis**: Jika file Excel sedang dibuka di aplikasi Microsoft Excel, log otomatis diselamatkan ke `data/log_plat_backup_YYYY-MM-DD.csv`.
- **Dual Snapshot Storage**: Menyimpan citra berwarna asli di `storage/snapshots/` dan versi *grayscale/preprocessed* di `storage/snapshots_grayscale/`.
- **Clickable Terminal Link**: Menampilkan link `file:///` di terminal untuk membuka hasil crop secara instant (`Ctrl + Click`).
- **Font Khusus Plat Nomor**: Overlay pada video menggunakan font khusus plat nomor Indonesia (`assets/fonts/PlatNomor-WyVnn.ttf`).
- **Pipeline Dataset & Training Custom**: Script generator dataset (`prepare_snapshot_dataset.py`) dan pelatih CNN (`train_character.py`).

---

## Struktur Direktori

```text
d:\PLAT\
├── main.py                     # Script utama ANPR & Gate Logger
├── prepare_snapshot_dataset.py   # Generator dataset karakter dari snapshot CCTV
├── train_character.py          # Script pelatihan CNN (PlateCharNetV2)
├── README.md                   # Dokumentasi proyek
│
├── models/                     # Bobot Model ML
│   ├── plate.pt                # Model YOLOv8 deteksi plat nomor
│   ├── char_model.pth          # Model PyTorch CNN karakter (opsional)
│   └── char_labels.json        # Mapping label kelas karakter
│
├── assets/                     # Font & Aset Visual
│   └── fonts/                  # Font khusus Plat Nomor Indonesia
│
├── data/                       # Log Spreadsheet Excel & Backup CSV
├── storage/                    # Simpanan Foto Snapshot Bukti
│   ├── snapshots/              # Image crop berwarna (Original)
│   └── snapshots_grayscale/    # Image crop grayscale (Preprocessed)
│
└── scripts/                    # Helper scripts (RTSP, DroidCam, List Devices)
```

---

## Panduan Penggunaan (Quickstart)

### 1. Install Dependensi
Pastikan Python 3.10+ sudah terinstall:

```bash
pip install ultralytics easyocr openpyxl opencv-python pillow numpy torch torchvision
```

### 2. Jalankan Program Utama
Jalankan aplikasi dengan kamera default/stream bawaan:

```bash
python main.py
```

### 3. Contoh Argumen Sumber Video

- **File Video Local**:
  ```bash
  python main.py --source "video/test_gate.mp4" --name "Gerbang-Depan"
  ```
- **Webcam (Device Index)**:
  ```bash
  python main.py --source 0 --name "Webcam-Laptop"
  ```
- **IP Camera / RTSP / HLS Stream**:
  ```bash
  python main.py --source "rtsp://admin:pass@192.168.1.100:554/stream1" --name "CCTV-Gate-1"
  python main.py --source "https://cctv.jogjaprov.go.id/cctv-proxy/atcs-kota/stream.m3u8" --name "CCTV-Malioboro"
  ```

---

## Panduan Koneksi DroidCam (Kamera HP)

Anda dapat menggunakan smartphone Android/iOS sebagai kamera ANPR menggunakan **DroidCam**:

### Mode 1: USB / Virtual Webcam (Sangat Direkomendasikan & Stabil)
1. Install **DroidCam Client** di Windows dan App DroidCam di HP.
2. Hubungkan HP via USB (aktifkan USB Debugging) atau via WiFi pada DroidCam Client PC.
3. Klik **Start** pada DroidCam Client PC.
4. Jalankan script menggunakan device index webcam Windows:
   ```powershell
   python main.py --source 0 --name "DroidCam-USB"
   ```
   *(Jika `--source 0` membuka webcam internal laptop, ganti ke `--source 1` atau `--source 2`)*.

### Mode 2: WiFi Stream / HTTP IP
1. Sambungkan HP dan Laptop ke jaringan WiFi yang sama.
2. Buka aplikasi DroidCam di HP dan perhatikan **WiFi IP** (misal: `192.168.82.42`).
3. *(Penting)* **Stop Streaming / tutup DroidCam Client di PC** terlebih dahulu agar port `4747` tidak di-lock oleh aplikasi Windows Client.
4. Jalankan program dengan URL video DroidCam:
   ```powershell
   python main.py --source "http://192.168.82.42:4747/video" --name "DroidCam-WiFi"
   ```

### Troubleshoot Jika DroidCam Gagal Terhubung:
Jika koneksi HTTP terputus atau gagal terhubung:
1. **Coba endpoint alternatif `/mjpegfeed`**:
   ```powershell
   python main.py --source "http://192.168.82.42:4747/mjpegfeed" --name "DroidCam-WiFi"
   ```
2. **Gunakan Mode Webcam Index (`--source 0` / `--source 1`)** jika DroidCam Client di Windows sedang dalam posisi aktif:
   ```powershell
   python main.py --source 0
   # atau jika webcam laptop aktif di index 0, gunakan index 1:
   python main.py --source 1
   ```
3. **Mekanisme Automatic Fallback**: `main.py` sudah diperbarui agar saat terjadi kegagalan membaca frame/stream, sistem akan otomatis melakukan reconnect ke kamera DroidCam Anda (mencoba berganti antara `/video` dan `/mjpegfeed`) tanpa pernah dialihkan secara sepihak ke CCTV publik.

---

## Parameter Konfigurasi Penting (`main.py`)

Anda dapat menyesuaikan beberapa batas ambang (threshold) di bagian awal file `main.py`:

```python
DET_CONF = 0.35              # Confidence threshold deteksi YOLOv8
MIN_OCR_CONF = 0.55          # Minimum confidence OCR untuk dikumpulkan ke voting
DEBOUNCE_SECONDS = 20        # Jeda detik pencegahan pencatatan ganda plat yang sama
OCR_EVERY_N_FRAMES = 2       # Frekuensi OCR (dilakukan setiap N frame)
```

---

## Dataset & Pelatihan Model Karakter Custom

1. **Ekstrak Karakter dari Snapshot**:
   ```bash
   python prepare_snapshot_dataset.py
   ```
2. **Latih Model PyTorch CNN**:
   ```bash
   python train_character.py
   ```
   *Hasil pelatihan akan disimpan otomatis ke `models/char_model.pth` dan `models/char_labels.json`.*

---

## Lisensi
Proyek ini dirilis di bawah [Lisensi MIT](LICENSE). Bebas digunakan dan dikembangkan untuk keperluan riset maupun komersial otomatisasi lalu lintas di Indonesia.
