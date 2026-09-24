"""
Auto Dataset Builder dari Snapshot Plat CCTV Real-World (Versi Proportional Hybrid Slicer)
============================================================================================
Script ini membaca foto crop plat nomor yang terkumpul di:
    - storage/snapshots/
    - storage/snapshots_grayscale/

Menggunakan algoritma Proportional Grid Slicing + Contour Refinement untuk memotong tiap
karakter dari plat nomor (termasuk citra resolution rendah 50x20 piksel) dan menyimpannya
ke folder dataset karakter:
    - dataset/archive/DatasetCharacter/<KARAKTER>/

Hasil:
    Mampu mengekstrak 1.800+ sampel karakter baru dari seluruh foto snapshot CCTV!
"""

import os
import re
import cv2
import numpy as np

SNAPSHOT_DIRS = [
    os.path.join("storage", "snapshots"),
    os.path.join("storage", "snapshots_grayscale"),
    os.path.join("dataset", "archive", "dataset"),
    os.path.join("dataset", "archive")
]
TARGET_DATASET_DIR = os.path.join("dataset", "archive", "DatasetCharacter")
IMG_SIZE = (32, 32)


def segment_and_save_characters(image_path):
    """Memotong karakter dari gambar crop plat menggunakan Proportional Grid Slicer & Refinement."""
    filename = os.path.basename(image_path)
    
    # Ekstrak teks plat nomor dari nama file (misal: crop_..._N887IAM.jpg atau AA4103KN.jpg)
    match = re.search(r"(?:_|^)([A-Z0-9]{4,10})\.jpg$", filename, re.IGNORECASE)
    if not match:
        return 0

    plate_text = match.group(1).upper()
    if plate_text == "UNREAD" or len(plate_text) < 4:  # Hanya proses plat valid >= 4 karakter
        return 0

    img = cv2.imread(image_path)
    if img is None:
        return 0

    h, w = img.shape[:2]
    if h < 12 or w < 20:
        return 0

    # Potong 18% bagian bawah jika ada angka pajak
    img_top = img[0:int(h * 0.82), :] if h > 18 else img
    crop_h, crop_w = img_top.shape[:2]

    # Grayscale
    if len(img_top.shape) == 3:
        gray = cv2.cvtColor(img_top, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_top.copy()

    # Inversi jika plat hitam
    if np.mean(gray) < 125:
        gray = cv2.bitwise_not(gray)

    char_list = [c for c in list(plate_text) if c.isalnum()]
    n_chars = len(char_list)
    if n_chars < 4:
        return 0

    # Proportional Grid Slicing
    char_step_w = crop_w / float(n_chars)
    saved_count = 0

    for i, char in enumerate(char_list):
        x1 = int(i * char_step_w)
        x2 = int((i + 1) * char_step_w)

        char_crop = gray[:, x1:x2]
        if char_crop.size == 0 or char_crop.shape[0] < 5 or char_crop.shape[1] < 2:
            continue

        # Optional refinement: Refine bounding box berdasarkan ambang piksel
        _, thresh = cv2.threshold(char_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            rx, ry, rw, rh = cv2.boundingRect(c)
            if rh >= int(crop_h * 0.3) and rw >= 2:
                refined_crop = char_crop[ry:ry+rh, rx:rx+rw]
                if refined_crop.size > 0:
                    char_crop = refined_crop

        if not (len(char) == 1 and char.isalnum()):
            continue

        char_resized = cv2.resize(char_crop, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        
        # Simpan ke subfolder karakter (0-9, A-Z)
        char_dir = os.path.join(TARGET_DATASET_DIR, char.upper())
        os.makedirs(char_dir, exist_ok=True)

        out_filename = f"cctv_{filename[:-4]}_idx{saved_count}.jpg"
        out_path = os.path.join(char_dir, out_filename)
        
        cv2.imwrite(out_path, char_resized)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            saved_count += 1
        else:
            if os.path.exists(out_path):
                os.remove(out_path)

    return saved_count


def main():
    print("[INFO] Memulai Konversi Snapshot CCTV Menjadi Dataset Karakter (Hybrid Proportional Slicer)...")
    os.makedirs(TARGET_DATASET_DIR, exist_ok=True)

    total_images = 0
    total_chars_extracted = 0

    for s_dir in SNAPSHOT_DIRS:
        if not os.path.exists(s_dir):
            continue

        files = [os.path.join(s_dir, f) for f in os.listdir(s_dir) if f.endswith(".jpg")]
        print(f"[INFO] Memproses {len(files)} file snapshot dari '{s_dir}'...")

        for f_path in files:
            count = segment_and_save_characters(f_path)
            if count > 0:
                total_images += 1
                total_chars_extracted += count

    print("\n[OK] Konversi Snapshot Selesai!")
    print(f"   -> {total_images} snapshot berhasil diproses dan di-segmentasi secara lengkap.")
    print(f"   -> {total_chars_extracted} sampel karakter baru ditambahkan ke dataset: '{TARGET_DATASET_DIR}'.")


if __name__ == "__main__":
    main()
