import os
# Redam warning OpenCV/FFmpeg
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"

import cv2
cv2.setLogLevel(0)

import re
import time
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO
import easyocr
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

# ============================ KONFIGURASI ============================
MODEL_PATH = os.path.join("models", "plate.pt") if os.path.exists(os.path.join("models", "plate.pt")) else ("plate.pt" if os.path.exists("plate.pt") else "yolov8n.pt")
SNAPSHOT_DIR = os.path.join("storage", "snapshots")

DET_CONF = 0.35              # Threshold deteksi YOLO
MIN_OCR_CONF = 0.50          # Threshold minimum confidence OCR
MIN_PLATE_WIDTH = 40         # Lebar minimum crop plat (piksel)
OCR_EVERY_N_FRAMES = 2       # Lakukan OCR tiap N frame per objek
DEBOUNCE_SECONDS = 15       # Mencegah logging ganda plat yang sama
# =====================================================================

# Peta Karakter Konfusi Plat Indonesia
CHAR_TO_DIGIT = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8'}
DIGIT_TO_CHAR = {'0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}

VALID_REGION_CODES = {
    "A", "B", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T", "W", "Z",
    "AA", "AB", "AD", "AE", "AG", "BA", "BB", "BD", "BE", "BG", "BH", "BK", "BL", "BM", "BN", "BP",
    "CC", "CD", "DA", "DB", "DC", "DD", "DE", "DG", "DH", "DK", "DL", "DM", "DN", "DR", "DS", "DT", "DW",
    "EA", "EB", "ED", "KB", "KH", "KT", "KU", "PA", "PB"
}


def clean_plate(raw_str: str):
    """Pembersihan & validasi format plat nomor Indonesia (Awalan 1-2 huruf, Tengah 1-4 angka, Akhiran 0-3 huruf)."""
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
    """Pra-pemrosesan citra plat nomor sebelum masuk ke EasyOCR."""
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    # Potong 20% bagian bawah plat (membuang garis angka bulan/tahun pajak)
    if h > 18:
        crop_top = crop[0:int(h * 0.80), :]
    else:
        crop_top = crop

    # Upscale gambar berwarna agar EasyOCR membaca lebih presisi
    min_height = 64
    scale = max(2.0, min_height / max(crop_top.shape[0], 1))
    upscaled = cv2.resize(crop_top, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return upscaled


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


def save_to_excel(timestamp, plate, conf, camera_name):
    """Log hasil pengenalan plat ke file Excel harian."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("data", exist_ok=True)
    excel_path = os.path.join("data", f"log_plat_{date_str}.xlsx")

    conf_str = f"{conf * 100:.1f}%" if conf > 0 else "N/A"

    if not os.path.exists(excel_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = f"Log {date_str}"
        headers = ["No", "Tanggal & Waktu", "Plat Nomor", "Confidence OCR", "Lokasi / Kamera"]
        ws.append(headers)
        header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        for col_num in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_num)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
    else:
        wb = openpyxl.load_workbook(excel_path)
        ws = wb.active

    row_idx = ws.max_row + 1
    no = row_idx - 1
    ws.append([no, timestamp, plate, conf_str, camera_name])

    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max(len(str(cell.value or '')) for cell in col)
        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

    try:
        wb.save(excel_path)
        print(f"[LOG EXCEL] {timestamp} | Plat: {plate} | Confidence: {conf_str}")
    except PermissionError:
        print(f"[WARN] File {excel_path} sedang dibuka di Microsoft Excel!")


def main():
    parser = argparse.ArgumentParser(description="ANPR System: YOLOv8 + ByteTrack + EasyOCR")
    parser.add_argument("--source", type=str, default="0", help="Path file video atau angka webcam (default: 0)")
    parser.add_argument("--name", type=str, default="Kamera 1", help="Nama lokasi / kamera")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    camera_name = args.name

    print(f"[1/3] Memuat Model YOLOv8 ({MODEL_PATH})...")
    yolo_model = YOLO(MODEL_PATH)

    print("[2/3] Memuat Engine EasyOCR...")
    ocr_reader = easyocr.Reader(["en"], gpu=False)

    print(f"[3/3] Membuka Input Video: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka sumber video: {source}")
        return

    # Tracking & State Storage
    track_readings = defaultdict(list)    # track_id -> [(text, conf)]
    track_last_seen = {}                   # track_id -> frame_idx
    logged_ids = set()                     # Set track_id yang sudah dicatat
    last_logged_plate_time = {}            # plate_text -> last_time

    # Load Font Khusus Plat Nomor
    plate_font, header_font, sub_font = load_custom_fonts()

    print("\n[INFO] Sistem Berjalan. Tekan 'q' pada jendela video untuk keluar.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video selesai / stream terputus.")
                break

            frame_idx += 1
            h, w = frame.shape[:2]

            # ---------------- 1. DETEKSI & TRACKING (YOLOv8 + ByteTrack) ----------------
            results = yolo_model.track(
                frame, persist=True, conf=DET_CONF, tracker="bytetrack.yaml", verbose=False
            )[0]

            current_frame_ids = set()
            annotated_frame = frame.copy()
            pil_img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results.boxes.id.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), tid in zip(boxes, track_ids):
                    current_frame_ids.add(tid)
                    track_last_seen[tid] = frame_idx

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop_w, crop_h = x2 - x1, y2 - y1

                    # ---------------- 2. PEMBACAAN OCR (EasyOCR) ----------------
                    if crop_w >= MIN_PLATE_WIDTH and frame_idx % OCR_EVERY_N_FRAMES == 0:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            processed_crop = preprocess_crop(crop)
                            ocr_results = ocr_reader.readtext(
                                processed_crop,
                                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
                                detail=1,
                                paragraph=False
                            )

                            if ocr_results:
                                raw_text = "".join(r[1] for r in ocr_results)
                                conf = sum(r[2] for r in ocr_results) / len(ocr_results)
                                cleaned_plate = clean_plate(raw_text)

                                if cleaned_plate and conf >= MIN_OCR_CONF:
                                    track_readings[tid].append((cleaned_plate, conf))

                    # ---------------- 3. VISUALISASI BOUNDING BOX & OVERLAY ----------------
                    readings = track_readings.get(tid, [])
                    display_text = readings[-1][0] if readings else f"TRACK #{tid}"

                    # Gambar Bounding Box
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

                    # Label Badge Teks Presisi
                    bbox = plate_font.getbbox(display_text)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    px, py = 12, 6

                    bx1 = x1
                    by1 = max(5, y1 - th - (py * 2) - 8)
                    bx2 = bx1 + tw + (px * 2)
                    by2 = by1 + th + (py * 2)

                    draw.rectangle([bx1, by1, bx2, by2], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
                    draw.text((bx1 + px, by1 + py - 2), display_text, font=plate_font, fill=(0, 0, 0))

            # ---------------- 4. VOTING & FINAL LOGGING SETELAH OBJEK MELINTAS ----------------
            for tid in list(track_readings.keys()):
                if tid in logged_ids:
                    continue

                is_active = tid in current_frame_ids
                frames_since_last_seen = frame_idx - track_last_seen.get(tid, frame_idx)

                # Jika objek tidak terdeteksi lagi di frame selama >= 4 frame (selesai melintas)
                if not is_active and frames_since_last_seen >= 4:
                    logged_ids.add(tid)
                    readings = track_readings[tid]

                    if readings:
                        # Voting hasil pembacaan terbanyak & rata-rata confidence
                        plate_scores = defaultdict(float)
                        plate_counts = defaultdict(int)

                        for text, conf in readings:
                            plate_scores[text] += conf
                            plate_counts[text] += 1

                        best_plate = max(plate_scores.keys(), key=lambda p: plate_scores[p] / plate_counts[p])
                        avg_conf = plate_scores[best_plate] / plate_counts[best_plate]

                        now_time = time.time()
                        if now_time - last_logged_plate_time.get(best_plate, 0) > DEBOUNCE_SECONDS:
                            last_logged_plate_time[best_plate] = now_time
                            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            save_to_excel(now_str, best_plate, avg_conf, camera_name)

            # ---------------- PANEL INFO PLAT NOMOR KIRI ATAS ----------------
            latest_plate_text = "-"
            if current_frame_ids:
                for active_tid in current_frame_ids:
                    r_list = track_readings.get(active_tid, [])
                    if r_list:
                        latest_plate_text = r_list[-1][0]
                        break

            now_ts = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            label_title = "Plat Nomor: "
            val_title = latest_plate_text
            sub_info_text = f"Kamera: {camera_name}  |  Waktu: {now_ts}"

            b_label = plate_font.getbbox(label_title)
            b_val = plate_font.getbbox(val_title)
            b_sub = sub_font.getbbox(sub_info_text)

            w_title = (b_label[2] - b_label[0]) + (b_val[2] - b_val[0])
            h_title = max(b_label[3] - b_label[1], b_val[3] - b_val[1])
            w_sub, h_sub = b_sub[2] - b_sub[0], b_sub[3] - b_sub[1]

            card_w = max(w_title, w_sub) + 30
            card_h = h_title + h_sub + 22
            card_x1, card_y1 = 15, 15
            card_x2, card_y2 = card_x1 + card_w, card_y1 + card_h

            draw.rectangle([card_x1, card_y1, card_x2, card_y2], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            draw.text((card_x1 + 15, card_y1 + 6), label_title, font=plate_font, fill=(0, 0, 0))
            
            x_val_pos = card_x1 + 15 + (b_label[2] - b_label[0])
            val_color = (0, 102, 204) if latest_plate_text != "-" else (120, 120, 120)
            draw.text((x_val_pos, card_y1 + 6), val_title, font=plate_font, fill=val_color)
            draw.text((card_x1 + 15, card_y1 + 12 + h_title), sub_info_text, font=sub_font, fill=(80, 80, 80))

            annotated_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2.imshow("ANPR System (YOLOv8 + ByteTrack + EasyOCR)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
