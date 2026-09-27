import os
# Redam warning jaringan FFmpeg / H264 dari OpenCV agar output terminal bersih
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"

import cv2
cv2.setLogLevel(0)

import re
import time
import csv
import queue
import threading
import argparse
from collections import defaultdict
from datetime import datetime
import json
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO
import easyocr

# PyTorch (opsional - untuk model CNN karakter kustom)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch tidak tersedia. Menggunakan EasyOCR saja.")
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as OpenpyxlImage

# ============================ KONFIGURASI ============================
CCTV_NAME = "CCTV Malioboro"
CCTV_URL = "https://cctv.jogjaprov.go.id/cctv-proxy/atcs-kota/AhmadJazuli.stream/chunklist_w999204661.m3u8" # Simpang Ahmad Jazuli

# Path Model & Storage
MODEL_PATH = os.path.join("models", "plate.pt") if os.path.exists(os.path.join("models", "plate.pt")) else ("plate.pt" if os.path.exists("plate.pt") else "yolov8n.pt")
CHAR_MODEL_PATH = os.path.join("models", "char_model.pth")
CHAR_LABELS_PATH = os.path.join("models", "char_labels.json")
SNAPSHOT_DIR = os.path.join("storage", "snapshots")
SNAPSHOT_GRAY_DIR = os.path.join("storage", "snapshots_grayscale")
USE_CNN_OCR = False  # Matikan sementara CNN karena akurasinya masih rendah (26%)


def get_daily_excel_path():
    """Mendapatkan path file Excel log harian otomatis berdasarkan tanggal (misal: data/log_plat_2026-09-24.xlsx)."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("data", exist_ok=True)
    return os.path.join("data", f"log_plat_{date_str}.xlsx")


def get_daily_csv_backup_path():
    """Mendapatkan path CSV backup harian otomatis jika Excel sedang dikunci."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("data", exist_ok=True)
    return os.path.join("data", f"log_plat_backup_{date_str}.csv")

# Parameter ANPR & Voting
DEBOUNCE_SECONDS = 20       # Cegah pencatatan ganda kendaraan/plat sama
DET_CONF = 0.35              # Threshold deteksi YOLO
MIN_OCR_CONF = 0.55          # Minimum OCR confidence untuk masuk voting (dinaikkan agar tebakan asal tidak lolos)
MIN_READINGS_TO_LOG = 1      # Set ke 1 agar kendaraan yang hanya terdeteksi 1-2 frame tetap tersimpan snapshot & log-nya
MIN_PLATE_WIDTH = 40         # Lebar minimum crop plat (piksel) untuk OCR
MAX_TRACK_AGE_FRAMES = 100   # Frame tanpa deteksi sebelum data tracking dibersihkan dari RAM
BUFFER_FRAMES_AFTER_CROSS = 10 # Buffer frame kumpul OCR sebelum finalisasi log
OCR_EVERY_N_FRAMES = 2       # OCR dilakukan tiap N frame per objek untuk menghemat CPU
# =====================================================================


# ========================= MODEL CNN KARAKTER =========================
class ResidualBlock(nn.Module):
    """Blok residual untuk PlateCharNetV2."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class PlateCharNetV2(nn.Module):
    """Model CNN dengan Residual Connections untuk klasifikasi karakter plat."""
    def __init__(self, num_classes=36):
        super(PlateCharNetV2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResidualBlock(32), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResidualBlock(64), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResidualBlock(128), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_cnn_ocr_model():
    """Memuat model CNN karakter yang sudah dilatih."""
    if not TORCH_AVAILABLE or not USE_CNN_OCR:
        return None, None

    if not os.path.exists(CHAR_MODEL_PATH) or not os.path.exists(CHAR_LABELS_PATH):
        print("[INFO] Model CNN karakter tidak ditemukan. Menggunakan EasyOCR saja.")
        print(f"       Jalankan 'python prepare_snapshot_dataset.py' lalu 'python train_character.py' untuk membuat model.")
        return None, None

    try:
        with open(CHAR_LABELS_PATH, "r") as f:
            idx_to_label = json.load(f)
        num_classes = len(idx_to_label)

        device = torch.device("cpu")
        model = PlateCharNetV2(num_classes=num_classes).to(device)

        checkpoint = torch.load(CHAR_MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            val_acc = checkpoint.get('val_acc', 0)
            print(f"[INFO] Model CNN Karakter dimuat! (Val Acc: {val_acc*100:.1f}%, {num_classes} kelas)")
        else:
            model.load_state_dict(checkpoint)
            print(f"[INFO] Model CNN Karakter dimuat! ({num_classes} kelas)")

        model.eval()
        return model, idx_to_label
    except Exception as e:
        print(f"[WARN] Gagal memuat model CNN: {e}")
        return None, None


def cnn_segment_and_recognize(processed_gray, cnn_model, idx_to_label):
    """
    Segmentasi karakter dari gambar crop plat yang sudah diproses,
    lalu kenali masing-masing karakter menggunakan model CNN.
    
    Returns: (plate_text, avg_confidence) atau (None, 0)
    """
    if cnn_model is None or processed_gray is None:
        return None, 0.0

    h, w = processed_gray.shape[:2]
    if h < 10 or w < 15:
        return None, 0.0

    # Binarisasi untuk segmentasi
    _, binary = cv2.threshold(processed_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Vertical projection profiling
    v_proj = np.sum(binary > 0, axis=0)
    threshold = h * 0.15

    in_char = False
    segments = []
    start = 0
    for x in range(w):
        if v_proj[x] > threshold and not in_char:
            in_char = True
            start = x
        elif v_proj[x] <= threshold and in_char:
            in_char = False
            if x - start >= 3:
                segments.append((start, x))
    if in_char and w - start >= 3:
        segments.append((start, w))

    # Filter segmen terlalu lebar
    refined = []
    for (sx, ex) in segments:
        seg_w = ex - sx
        if seg_w > w * 0.35:
            mid = sx + seg_w // 2
            refined.append((sx, mid))
            refined.append((mid, ex))
        else:
            refined.append((sx, ex))
    segments = refined

    if len(segments) < 3 or len(segments) > 12:
        return None, 0.0

    # Klasifikasi setiap segmen dengan CNN
    device = next(cnn_model.parameters()).device
    chars = []
    confs = []

    for (sx, ex) in segments:
        char_crop = processed_gray[:, sx:ex]
        if char_crop.size == 0 or char_crop.shape[1] < 2:
            continue

        # Contour refinement
        _, ct = cv2.threshold(char_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(ct, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            rx, ry, rw, rh = cv2.boundingRect(c)
            if rh >= int(h * 0.25) and rw >= 2:
                char_crop = char_crop[ry:ry+rh, rx:rx+rw]

        # Resize ke 32x32 dan normalisasi
        char_resized = cv2.resize(char_crop, (32, 32), interpolation=cv2.INTER_CUBIC)
        tensor = torch.from_numpy(char_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5  # Normalisasi seperti saat training
        tensor = tensor.to(device)

        with torch.no_grad():
            output = cnn_model(tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        pred_label = idx_to_label.get(str(predicted_idx.item()), "?")
        conf_val = confidence.item()

        if conf_val >= 0.3:  # Minimum confidence per karakter
            chars.append(pred_label)
            confs.append(conf_val)

    if len(chars) < 3:
        return None, 0.0

    plate_text = "".join(chars)
    avg_conf = sum(confs) / len(confs)

    return plate_text, avg_conf
# =====================================================================

# Confusable mapping untuk plat nomor Indonesia
CHAR_TO_DIGIT = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8'}
DIGIT_TO_CHAR = {'0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}

VALID_REGION_CODES = {
    # 1 Huruf
    "A", "B", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T", "W", "Z",
    # 2 Huruf (Jawa & Sumatra)
    "AA", "AB", "AD", "AE", "AG", "BA", "BB", "BD", "BE", "BG", "BH", "BK", "BL", "BM", "BN", "BP",
    # Korps Konsul & Diplomatik
    "CC", "CD",
    # Kalimantan, Sulawesi, Maluku, Bali, Nusa Tenggara, Papua
    "DA", "DB", "DC", "DD", "DE", "DG", "DH", "DK", "DL", "DM", "DN", "DR", "DS", "DT", "DW",
    "EA", "EB", "ED", "KB", "KH", "KT", "KU", "PA", "PB"
}


def load_custom_fonts():
    """Memuat font plat nomor khusus dari folder assets/fonts/."""
    font_paths = [
        os.path.join("assets", "fonts", "PlatNomor-WyVnn.ttf"),
        os.path.join("assets", "fonts", "PlatNomor-eZ2dm.otf"),
        os.path.join("plat-nomor-font", "PlatNomor-WyVnn.ttf"),
        os.path.join("plat-nomor-font", "PlatNomor-eZ2dm.otf"),
    ]
    font_path = next((p for p in font_paths if os.path.exists(p)), None)

    if font_path:
        try:
            plate_font = ImageFont.truetype(font_path, 28)
            header_font = ImageFont.truetype(font_path, 22)
            sub_font = ImageFont.truetype(font_path, 16)
            print(f"[INFO] Berhasil memuat Font Plat Nomor: {font_path}")
            return plate_font, header_font, sub_font
        except Exception as e:
            print(f"[WARN] Gagal memuat font {font_path}: {e}")

    default = ImageFont.load_default()
    return default, default, default


def clean_plate(raw_str: str):
    """Pembersihan & validasi format plat nomor Indonesia (Awalan 1-2 huruf, Tengah 1-4 angka, Akhiran 0-3 huruf opsional)."""
    if not raw_str:
        return None
    raw_str = raw_str.strip().upper()
    raw_clean = re.sub(r"[^A-Z0-9 ]", "", raw_str)
    
    def try_assemble_plate(p_raw, n_raw, s_raw=""):
        p_cand = "".join(DIGIT_TO_CHAR.get(c, c) for c in p_raw)
        if not (1 <= len(p_cand) <= 2 and p_cand in VALID_REGION_CODES):
            return None

        n_cand = "".join(CHAR_TO_DIGIT.get(c, c) for c in n_raw)
        if not (n_cand.isdigit() and 1 <= len(n_cand) <= 4):
            return None

        s_cand = "".join(DIGIT_TO_CHAR.get(c, c) for c in s_raw) if s_raw else ""
        if len(s_cand) > 3 or (len(s_cand) > 0 and not s_cand.isalpha()):
            return None

        return f"{p_cand} {n_cand} {s_cand}".strip()

    parts = raw_clean.split()
    if len(parts) >= 3:
        res = try_assemble_plate(parts[0], parts[1], parts[2])
        if res:
            return res
    if len(parts) >= 2:
        p_raw, remainder = parts[0], parts[1]
        res = try_assemble_plate(p_raw, remainder, "")
        if res:
            return res
        for n_len in range(1, min(5, len(remainder) + 1)):
            n_raw = remainder[:n_len]
            s_raw = remainder[n_len:]
            res = try_assemble_plate(p_raw, n_raw, s_raw)
            if res:
                return res

    text = re.sub(r"[^A-Z0-9]", "", raw_clean)
    for p_len in (1, 2):
        if len(text) >= p_len:
            p_raw = text[:p_len]
            remainder = text[p_len:]
            res = try_assemble_plate(p_raw, remainder, "")
            if res:
                return res
            for n_len in range(1, min(5, len(remainder) + 1)):
                n_raw = remainder[:n_len]
                s_raw = remainder[n_len:]
                res = try_assemble_plate(p_raw, n_raw, s_raw)
                if res:
                    return res

    return None


def preprocess_crop(crop):
    """Pra-pemrosesan citra plat nomor untuk EasyOCR.
    
    PENTING: EasyOCR memiliki pipeline preprocessing internal yang sangat canggih.
    Memberikan gambar BERWARNA yang sudah di-upscale menghasilkan akurasi JAUH LEBIH BAIK
    daripada gambar biner/threshold/CLAHE.
    """
    if crop is None or crop.size == 0:
        return crop
        
    h, w = crop.shape[:2]
    # 1. Potong 20% bagian bawah plat untuk membuang baris angka bulan/tahun pajak
    if h > 18:
        crop_top = crop[0:int(h * 0.80), :]
    else:
        crop_top = crop

    # 2. Upscale gambar BERWARNA (EasyOCR lebih akurat dengan gambar natural)
    min_height = 64  # EasyOCR butuh minimal ~64px tinggi untuk akurasi bagus
    scale = max(2.0, min_height / max(crop_top.shape[0], 1))
    upscaled = cv2.resize(crop_top, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    return upscaled

class AsyncOCRWorker:
    """Worker pool berbasis Queue Multithreading untuk pemrosesan Hybrid CNN+EasyOCR."""

    def __init__(self, reader, readings, best_crops, lock, cnn_model=None, idx_to_label=None):
        self.reader = reader
        self.readings = readings
        self.best_crops = best_crops
        self.lock = lock
        self.cnn_model = cnn_model
        self.idx_to_label = idx_to_label
        self.task_queue = queue.Queue(maxsize=30)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def enqueue(self, tid, crop):
        if not self.task_queue.full():
            self.task_queue.put((tid, crop))

    def _process_result(self, tid, crop, plate_cleaned, ocr_conf):
        """Helper: simpan hasil OCR ke readings & best_crops."""
        if plate_cleaned and ocr_conf >= MIN_OCR_CONF:
            with self.lock:
                self.readings[tid].append((plate_cleaned, ocr_conf))
                crop_area = crop.shape[0] * crop.shape[1]
                crop_quality_score = (ocr_conf * 1000.0) + crop_area
                if tid not in self.best_crops or crop_quality_score > self.best_crops[tid][2]:
                    self.best_crops[tid] = (crop.copy(), preprocess_crop(crop).copy(), crop_quality_score)

    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                tid, crop = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            processed = preprocess_crop(crop)
            best_plate = None
            best_conf = 0.0

            # ---- Method 1: CNN Character Recognition ----
            if self.cnn_model is not None and self.idx_to_label is not None:
                cnn_text, cnn_conf = cnn_segment_and_recognize(processed, self.cnn_model, self.idx_to_label)
                if cnn_text:
                    cnn_cleaned = clean_plate(cnn_text)
                    if cnn_cleaned and cnn_conf >= MIN_OCR_CONF:
                        best_plate = cnn_cleaned
                        best_conf = cnn_conf

            # ---- Method 2: EasyOCR (fallback atau konfirmasi) ----
            # Kirim gambar BERWARNA (processed) langsung ke EasyOCR.
            # EasyOCR memiliki internal preprocessing (grayscale, binarization, dll) yang dioptimasi.
            ocr_res = self.reader.readtext(processed, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ", detail=1, paragraph=False)
            if ocr_res:
                text_raw = "".join(r[1] for r in ocr_res)
                ocr_conf = sum(r[2] for r in ocr_res) / len(ocr_res)
                ocr_cleaned = clean_plate(text_raw)

                if ocr_cleaned and ocr_conf >= MIN_OCR_CONF:
                    # Jika CNN tidak berhasil, gunakan EasyOCR
                    if best_plate is None:
                        best_plate = ocr_cleaned
                        best_conf = ocr_conf
                    # Jika keduanya berhasil, pilih yang confidence lebih tinggi
                    elif ocr_conf > best_conf:
                        best_plate = ocr_cleaned
                        best_conf = ocr_conf
                    # Jika CNN dan EasyOCR setuju, boost confidence
                    elif best_plate == ocr_cleaned:
                        best_conf = min(1.0, (best_conf + ocr_conf) / 2.0 + 0.05)

            # Simpan hasil terbaik
            if best_plate:
                self._process_result(tid, crop, best_plate, best_conf)

            self.task_queue.task_done()


def init_excel(file_path):
    """Inisialisasi file Excel khusus pengumpulan data crop plat nomor harian."""
    if not os.path.exists(file_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        date_str = datetime.now().strftime("%Y-%m-%d")
        ws.title = f"Log {date_str}"
        headers = ["No", "Tanggal & Waktu", "Plat Nomor", "Confidence OCR", "Kamera CCTV", "Gambar Crop Plat (Grayscale)", "Path Color", "Path Grayscale"]
        ws.append(headers)

        header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        for col_num in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_num)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        ws.column_dimensions["F"].width = 22  # Lebar khusus kolom Gambar Crop Plat
        try:
            wb.save(file_path)
        except PermissionError:
            pass


def save_to_excel(timestamp, plate, conf, camera, color_snap_path, gray_snap_path, file_path=None):
    """Menyimpan hasil deteksi ke Log Excel Harian & menempelkan Gambar Crop Plat Grayscale ke Kolom F."""
    if file_path is None:
        file_path = get_daily_excel_path()

    backup_csv_path = get_daily_csv_backup_path()
    conf_str = f"{conf * 100:.1f}%" if conf > 0 else "N/A"
    
    try:
        init_excel(file_path)
        wb = openpyxl.load_workbook(file_path)
        ws = wb.active

        row_idx = ws.max_row + 1
        no = row_idx - 1

        row_data = [no, timestamp, plate, conf_str, camera, "", color_snap_path, gray_snap_path]
        ws.append(row_data)

        # Set tinggi baris agar gambar fit dengan rapi
        ws.row_dimensions[row_idx].height = 42

        # Style data rows
        data_font = Font(name="Calibri", size=10)
        for col_num in range(1, len(row_data) + 1):
            cell = ws.cell(row=row_idx, column=col_num)
            cell.font = data_font
            if col_num in (1, 2, 4):
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif col_num in (3, 7, 8):
                cell.alignment = Alignment(horizontal="left", vertical="center")

        # ---------------- EMBEDDED GAMBAR CROP PLAT GRAYSCALE KE SEL EXCEL (KOLOM F) ----------------
        if gray_snap_path and os.path.exists(gray_snap_path):
            try:
                img_embed = OpenpyxlImage(gray_snap_path)
                img_embed.width = 130
                img_embed.height = 42
                ws.add_image(img_embed, f"F{row_idx}")
            except Exception as img_err:
                print(f"[WARN IMAGE EMBED] Gagal menempel gambar ke Excel: {img_err}")

        # Dynamic Column Widths
        for col in ws.columns:
            col_letter = get_column_letter(col[0].column)
            if col_letter == "F":
                ws.column_dimensions["F"].width = 22
            else:
                max_len = max(len(str(cell.value or '')) for cell in col)
                ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

        wb.save(file_path)

        # Format Clickable Link untuk Terminal (Tekan Ctrl + Click di terminal untuk membuka gambar)
        abs_gray_path = os.path.abspath(gray_snap_path).replace("\\", "/") if gray_snap_path else ""
        abs_color_path = os.path.abspath(color_snap_path).replace("\\", "/") if color_snap_path else ""
        clickable_link_gray = f"file:///{abs_gray_path}" if abs_gray_path else ""
        clickable_link_color = f"file:///{abs_color_path}" if abs_color_path else ""

        today_date = datetime.now().strftime("%Y-%m-%d")
        print(f"[OK] [LOG HARIAN {today_date}] {timestamp} | Plat: {plate} | Conf: {conf_str} | Open Gray: {clickable_link_gray} | Color: {clickable_link_color}")
        return True

    except PermissionError:
        print(f"[WARN PERMISSION] File Excel '{file_path}' sedang terbuka di Microsoft Excel!")
        print("   --> Mohon tutup aplikasi Excel agar data & gambar tersimpan langsung ke file .xlsx.")
        
        file_exists = os.path.exists(backup_csv_path)
        with open(backup_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Plat Nomor", "Confidence", "Kamera", "ColorSnap", "GraySnap"])
            writer.writerow([timestamp, plate, conf_str, camera, color_snap_path, gray_snap_path])
        print(f"  [BACKUP CSV HARIAN] Data diselamatkan ke: {backup_csv_path}")
        return False
        
    except Exception as e:
        print(f"[WARN SAVE ERROR] Gagal menyimpan ke Excel: {e}")
        return False
        
    except Exception as e:
        print(f"[WARN SAVE ERROR] Gagal menyimpan ke Excel: {e}")
        return False


def save_snapshots(frame, crop_color, crop_gray, plate):
    """Menyimpan citra crop plat nomor presisi ke folder snapshots/ (Warna) dan snapshots_grayscale/ (Grayscale)."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_GRAY_DIR, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    safe_plate = plate.replace(" ", "") if plate else "UNREAD"
    
    crop_filename = f"crop_{ts}_{safe_plate}.jpg"
    color_path = os.path.join(SNAPSHOT_DIR, crop_filename)
    gray_path = os.path.join(SNAPSHOT_GRAY_DIR, f"crop_gray_{ts}_{safe_plate}.jpg")
    
    if crop_color is not None and crop_color.size > 0:
        cv2.imwrite(color_path, crop_color)
    else:
        cv2.imwrite(color_path, frame)

    if crop_gray is not None and crop_gray.size > 0:
        cv2.imwrite(gray_path, crop_gray)
    else:
        if crop_color is not None and crop_color.size > 0:
            gray_img = cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(gray_path, gray_img)
        else:
            cv2.imwrite(gray_path, frame)

    return color_path, gray_path


def main():
    parser = argparse.ArgumentParser(description="High-Speed Indonesian ANPR System")
    parser.add_argument("--source", type=str, default=CCTV_URL, help="Path ke file video, IP Camera (RTSP/HTTP), atau angka (0 untuk webcam)")
    parser.add_argument("--name", type=str, default=CCTV_NAME, help="Nama kamera/lokasi CCTV untuk keperluan logging")
    args = parser.parse_args()

    stream_url = int(args.source) if args.source.isdigit() else args.source
    camera_name = args.name

    print(f"[INFO] Memuat Model YOLO ({MODEL_PATH})...")
    model = YOLO(MODEL_PATH)
    
    print("[INFO] Memuat Engine EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=False)

    # ---- Load Model CNN Karakter (opsional, meningkatkan akurasi) ----
    cnn_model, idx_to_label = load_cnn_ocr_model()
    if cnn_model is not None:
        print("[INFO] Mode Hybrid CNN + EasyOCR AKTIF (akurasi maksimal)")
    else:
        print("[INFO] Mode EasyOCR saja (tanpa CNN)")

    print(f"[INFO] Menghubungkan ke CCTV: {camera_name}")
    print(f"[INFO] Stream URL: {stream_url}")
    cap = cv2.VideoCapture(stream_url)

    # Load Font Khusus Plat Nomor
    plate_font, header_font, sub_font = load_custom_fonts()
    init_excel(get_daily_excel_path())

    # ---------------- STATE TRACKING & MULTI-THREADING QUEUE ----------------
    track_readings = defaultdict(list)       # tid -> [(plate, conf)]
    best_crops = {}                          # tid -> (crop_color, crop_gray, conf_score)
    track_first_seen = {}                    # tid -> frame_idx
    track_last_seen = {}                     # tid -> frame_idx
    logged_tids = set()                      # set tid yang sudah dicatat ke Excel
    last_logged_plate_time = {}              # plate_text -> timestamp
    readings_lock = threading.Lock()

    # Worker Thread Multithreading untuk Hybrid CNN+EasyOCR Async
    ocr_worker = AsyncOCRWorker(reader, track_readings, best_crops, readings_lock,
                                cnn_model=cnn_model, idx_to_label=idx_to_label)
    ocr_worker.start()
    # -------------------------------------------------------------------------

    print(f"[INFO] File Excel Log Harian Siap: {os.path.abspath(get_daily_excel_path())}")
    print("[INFO] High-Speed Async Hybrid OCR & ByteTrack Aktif. Tekan 'q' pada jendela video untuk keluar.\n")

    frame_idx = 0
    retry_count = 0
    max_retries = 5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                retry_count += 1
                print(f"[WARN] Gagal membaca stream ({retry_count}/{max_retries}). Reconnecting...")
                cap.release()
                time.sleep(2)
                # Coba reconnect ke sumber yang ditentukan user (bukan dialihkan ke CCTV_URL)
                cap = cv2.VideoCapture(stream_url)
                
                # Jika sumber berupa URL DroidCam HTTP dan gagal, coba variasi URL DroidCam lain (/video <-> /mjpegfeed)
                if not cap.isOpened() and isinstance(stream_url, str) and "4747" in stream_url:
                    alt_url = stream_url.replace("/video", "/mjpegfeed") if "/video" in stream_url else stream_url.replace("/mjpegfeed", "/video")
                    print(f"[INFO] Mencoba URL alternatif DroidCam: {alt_url}")
                    cap = cv2.VideoCapture(alt_url)
                    if cap.isOpened():
                        stream_url = alt_url

                if retry_count >= max_retries:
                    print("[ERROR] Koneksi kamera/stream terputus total. Program berhenti.")
                    break
                continue

            retry_count = 0
            frame_idx += 1
            h, w = frame.shape[:2]

            # 1. Objek Tracking menggunakan ByteTrack
            results = model.track(
                frame, persist=True, conf=DET_CONF, tracker="bytetrack.yaml", verbose=False
            )[0]
            
            current_frame_tids = set()
            annotated_frame = frame.copy()
            
            pil_img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                tids = results.boxes.id.cpu().numpy().astype(int)
                cls_ids = results.boxes.cls.cpu().numpy().astype(int)

                plate_parts = []
                for (x1, y1, x2, y2), tid, cls_id in zip(boxes, tids, cls_ids):
                    label = model.names.get(cls_id, str(cls_id)).lower()
                    is_plate_obj = "plate" in label or label == "0" or (cls_id == 0 and "plate" in MODEL_PATH.lower())
                    
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, w), min(y2, h)
                    
                    if is_plate_obj:
                        plate_parts.append({
                            'box': [x1, y1, x2, y2],
                            'tid': tid
                        })
                    else:
                        # Jika objek lain (misal kendaraan), gambar box standar
                        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

                # ========================================================
                # 1.B LOGIC PENGGABUNGAN KOTAK (MERGE BOXES)
                # ========================================================
                merged_plates = []
                for part in plate_parts:
                    px1, py1, px2, py2 = part['box']
                    ptid = part['tid']
                    merged = False
                    
                    for group in merged_plates:
                        gx1, gy1, gx2, gy2 = group['box']
                        
                        # Cek apakah part sejajar secara vertikal (sebaris)
                        y_overlap = max(0, min(py2, gy2) - max(py1, gy1))
                        h_part = py2 - py1
                        h_group = gy2 - gy1
                        
                        # Jika overlap vertikal minimal 30% dari tinggi huruf
                        if min(h_part, h_group) > 0 and (y_overlap / min(h_part, h_group)) > 0.3:
                            # Cek jarak horizontal agar huruf tidak terlalu jauh dari plat
                            x_dist = min(abs(px1 - gx2), abs(gx1 - px2))
                            max_w = max(gx2 - gx1, px2 - px1)
                            
                            # PERBAIKAN: Toleransi jarak diperbesar (4x) untuk mengatasi YOLO yang kadang meloncati bagian tengah
                            if x_dist < max_w * 4.0:
                                group['box'] = [min(px1, gx1), min(py1, gy1), max(px2, gx2), max(py2, gy2)]
                                group['tids'].append(ptid)
                                group['primary_tid'] = min(group['primary_tid'], ptid)
                                merged = True
                                break
                    
                    if not merged:
                        merged_plates.append({
                            'box': [px1, py1, px2, py2],
                            'tids': [ptid],
                            'primary_tid': ptid
                        })

                # ========================================================
                # 1.C PROSES KOTAK YANG SUDAH DIGABUNG KE OCR
                # ========================================================
                for group in merged_plates:
                    x1, y1, x2, y2 = group['box']
                    tid = group['primary_tid']  # Pakai ID terkecil sebagai ID plat utama
                    crop_w = x2 - x1
                    crop_h = y2 - y1

                    current_frame_tids.add(tid)
                    track_last_seen[tid] = frame_idx
                    if tid not in track_first_seen:
                        track_first_seen[tid] = frame_idx

                    # ---- Enqueue Task ke OCR Worker Thread secara Async ----
                    # Berikan margin sedikit (padding) 5 pixel agar crop lebih utuh
                    pad = 5
                    cy1, cy2 = max(0, y1-pad), min(h, y2+pad)
                    cx1, cx2 = max(0, x1-pad), min(w, x2+pad)
                    
                    if crop_w >= MIN_PLATE_WIDTH:
                        if frame_idx % OCR_EVERY_N_FRAMES == 0:
                            crop = frame[cy1:cy2, cx1:cx2]
                            if crop.size > 0:
                                ocr_worker.enqueue(tid, crop.copy())

                    # Dapatkan hasil OCR sementara untuk tampilan UI per kendaraan (#tid)
                    with readings_lock:
                        readings = list(track_readings.get(tid, []))
                    display_text = readings[-1][0] if readings else f"TRACK #{tid}"

                    # ---------------- RENDERING BADGE OVERLAY PER VEHICLE ----------------
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

                    bbox = plate_font.getbbox(display_text)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    px, py = 12, 6

                    bx1 = x1
                    by1 = max(5, y1 - th - (py * 2) - 8)
                    bx2 = bx1 + tw + (px * 2)
                    by2 = by1 + th + (py * 2)

                    draw.rectangle([bx1, by1, bx2, by2], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
                    draw.text((bx1 + px, by1 + py - 2), display_text, font=plate_font, fill=(0, 0, 0))

            # ---------------- MULTI-FRAME WEIGHTED CONFIDENCE VOTING ----------------
            for tid in list(track_readings.keys()):
                if tid in logged_tids:
                    continue

                with readings_lock:
                    readings = list(track_readings.get(tid, []))
                    best_crop_tuple = best_crops.get(tid, (None, None, 0.0))

                is_active = tid in current_frame_tids
                frames_since_last_seen = frame_idx - track_last_seen.get(tid, frame_idx)

                # Finalisasi HANYA setelah kendaraan tidak lagi terlihat di frame (telah selesai melintas)
                if not is_active and frames_since_last_seen >= 4:
                    logged_tids.add(tid)

                    if len(readings) >= MIN_READINGS_TO_LOG:
                        plate_scores = defaultdict(float)
                        plate_counts = defaultdict(int)

                        for p_text, conf_val in readings:
                            plate_scores[p_text] += conf_val
                            plate_counts[p_text] += 1

                        best_voted_plate = max(plate_scores.keys(), key=lambda p: plate_scores[p] / plate_counts[p])
                        avg_conf = plate_scores[best_voted_plate] / plate_counts[best_voted_plate]
                        
                        now_time = time.time()
                        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        if now_time - last_logged_plate_time.get(best_voted_plate, 0) > DEBOUNCE_SECONDS:
                            last_logged_plate_time[best_voted_plate] = now_time

                            best_crop_color = best_crop_tuple[0]
                            best_crop_gray = best_crop_tuple[1]
                            color_snap_path, gray_snap_path = save_snapshots(frame, best_crop_color, best_crop_gray, best_voted_plate)
                            save_to_excel(now_str, best_voted_plate, avg_conf, camera_name, color_snap_path, gray_snap_path)

            # ---------------- HOUSEKEEPING MEMORY LEAK ----------------
            if frame_idx % 100 == 0:
                stale_tids = [
                    t for t, last_f in track_last_seen.items()
                    if frame_idx - last_f > MAX_TRACK_AGE_FRAMES
                ]
                for t in stale_tids:
                    with readings_lock:
                        track_readings.pop(t, None)
                        best_crops.pop(t, None)
                    track_first_seen.pop(t, None)
                    track_last_seen.pop(t, None)
                    logged_tids.discard(t)

            # ---------------- HEADER PANEL OVERLAY ----------------
            now_ts = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            active_excel_log = get_daily_excel_path()
            top_text_1 = f"CCTV: {camera_name}"
            top_text_2 = f"Waktu: {now_ts}  |  Excel Log Harian: {active_excel_log}"

            b1 = header_font.getbbox(top_text_1)
            b2 = sub_font.getbbox(top_text_2)

            w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
            w2, h2 = b2[2] - b2[0], b2[3] - b2[1]

            card_w = max(w1, w2) + 30
            card_h = h1 + h2 + 22
            card_x1, card_y1 = 15, 15
            card_x2, card_y2 = card_x1 + card_w, card_y1 + card_h

            draw.rectangle([card_x1, card_y1, card_x2, card_y2], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            draw.text((card_x1 + 15, card_y1 + 6), top_text_1, font=header_font, fill=(0, 0, 0))
            draw.text((card_x1 + 15, card_y1 + 10 + h1), top_text_2, font=sub_font, fill=(60, 60, 60))

            annotated_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Preview GUI
            try:
                cv2.imshow("ANPR - Multi-Frame Voted CCTV", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Pengujian dihentikan oleh pengguna.")
                    break
            except cv2.error:
                pass

    finally:
        ocr_worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
