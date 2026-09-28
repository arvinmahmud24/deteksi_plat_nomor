import os
# Redam warning OpenCV/FFmpeg
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"

import cv2
cv2.setLogLevel(0)

import re
import argparse
from datetime import datetime
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO
import easyocr

# ============================ KONFIGURASI ============================
MODEL_PATH = os.path.join("models", "plate.pt") if os.path.exists(os.path.join("models", "plate.pt")) else ("plate.pt" if os.path.exists("plate.pt") else "yolov8n.pt")
DET_CONF = 0.35              # Threshold confidence deteksi YOLO
MIN_OCR_CONF = 0.40          # Threshold confidence minimum OCR
MIN_PLATE_WIDTH = 35         # Lebar minimum crop plat (piksel)
# =====================================================================

# Mapping karakter konfusi khas plat nomor Indonesia
CHAR_TO_DIGIT = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8'}
DIGIT_TO_CHAR = {'0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}

VALID_REGION_CODES = {
    "A", "B", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T", "W", "Z",
    "AA", "AB", "AD", "AE", "AG", "BA", "BB", "BD", "BE", "BG", "BH", "BK", "BL", "BM", "BN", "BP",
    "CC", "CD", "DA", "DB", "DC", "DD", "DE", "DG", "DH", "DK", "DL", "DM", "DN", "DR", "DS", "DT", "DW",
    "EA", "EB", "ED", "KB", "KH", "KT", "KU", "PA", "PB"
}


def clean_plate(raw_str: str):
    """Pembersihan & normalisasi format plat nomor Indonesia."""
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
    """Pra-pemrosesan citra plat nomor: Cropping, Upscaling, Grayscale, dan Thresholding Hitam-Putih (Binarisasi)."""
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    # 1. Potong 20% bagian bawah plat (membuang baris bulan/tahun pajak)
    if h > 18:
        crop_top = crop[0:int(h * 0.80), :]
    else:
        crop_top = crop

    # 2. Upscale citra plat agar karakter lebih besar dan jelas (minimal tinggi 64px)
    min_height = 64
    scale = max(2.0, min_height / max(crop_top.shape[0], 1))
    upscaled = cv2.resize(crop_top, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 3. Konversi ke Grayscale (Abu-abu)
    if len(upscaled.shape) == 3:
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    else:
        gray = upscaled.copy()

    # 4. Pengurangan Noise (Denoising)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 5. Thresholding Hitam-Putih (Otsu Binarization agar teks & background terpisah kontras)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Jika plat didominasi latar putih (plat baru), balikkan warna jika perlu agar teks selalu putih/hitam kontras
    # Namun EasyOCR umumnya bekerja sangat baik dengan format BGR atau Grayscale threshold 3-channel
    binary_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return binary_3ch


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


def main():
    parser = argparse.ArgumentParser(description="Simple ANPR: YOLOv8 Plate Detector + EasyOCR")
    parser.add_argument("--source", type=str, default="0", help="Path file video/gambar atau angka webcam (default: 0)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    print(f"[1/2] Memuat Model YOLOv8 ({MODEL_PATH})...")
    yolo_model = YOLO(MODEL_PATH)

    print("[2/2] Memuat Engine EasyOCR...")
    ocr_reader = easyocr.Reader(["en"], gpu=False)

    print(f"\n[INFO] Membuka Input Source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka sumber input: {source}")
        return

    # Load Font Khusus Plat Nomor
    plate_font, header_font, sub_font = load_custom_fonts()
    last_detected_plate_text = "-"
    print("[INFO] Sistem Berjalan. Tekan 'q' untuk keluar.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video/Stream selesai.")
                break

            h, w = frame.shape[:2]

            # ---------------- 1. DETEKSI LOKASI PLAT (YOLOv8 ONLY) ----------------
            results = yolo_model.predict(frame, conf=DET_CONF, verbose=False)[0]

            annotated_frame = frame.copy()
            pil_img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)

                for (x1, y1, x2, y2) in boxes:
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop_w, crop_h = x2 - x1, y2 - y1

                    display_text = "Plat Nomor"

                    # ---------------- 2. CROP, UPSCALE & EASYOCR ----------------
                    if crop_w >= MIN_PLATE_WIDTH:
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
                                raw_text = " ".join(r[1] for r in ocr_results)
                                conf = sum(r[2] for r in ocr_results) / len(ocr_results)
                                cleaned_plate = clean_plate(raw_text)

                                text_result = cleaned_plate
                                if not text_result and raw_text:
                                    # Fallback: jika clean_plate belum cocok, sisipkan spasi otomatis di antara huruf dan angka
                                    compact = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
                                    m = re.match(r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{0,3})$", compact)
                                    if m:
                                        text_result = f"{m.group(1)} {m.group(2)} {m.group(3)}".strip()
                                    else:
                                        text_result = raw_text

                                if conf >= MIN_OCR_CONF and text_result:
                                    display_text = f"{text_result} ({conf*100:.0f}%)"

                    # ---------------- 3. RENDERING BOX & OVERLAY TEKS ----------------
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

                    # Label Badge Teks Presisi (Kustom Font Plat Nomor)
                    bbox = plate_font.getbbox(display_text)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    px, py = 12, 6

                    bx1 = x1
                    by1 = max(5, y1 - th - (py * 2) - 8)
                    bx2 = bx1 + tw + (px * 2)
                    by2 = by1 + th + (py * 2)

                    # Simpan hasil teks plat untuk panel kiri atas
                    if display_text != "Plat Nomor":
                        last_detected_plate_text = display_text.split(" (")[0]

            # ---------------- PANEL INFO PLAT NOMOR KIRI ATAS ----------------
            now_ts = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            label_title = "Plat Nomor: "
            val_title = last_detected_plate_text
            sub_info_text = f"Mode: Real-Time B&W OCR  |  Waktu: {now_ts}"

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
            val_color = (0, 102, 204) if last_detected_plate_text != "-" else (120, 120, 120)
            draw.text((x_val_pos, card_y1 + 6), val_title, font=plate_font, fill=val_color)
            draw.text((card_x1 + 15, card_y1 + 12 + h_title), sub_info_text, font=sub_font, fill=(80, 80, 80))

            annotated_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2.imshow("Simple ANPR (YOLOv8 + EasyOCR)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
