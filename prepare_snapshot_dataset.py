"""
Auto Dataset Builder dari Snapshot Plat CCTV Real-World (Versi Advanced)
========================================================================
Script ini membaca foto crop plat nomor yang terkumpul di:
    - storage/snapshots/
    - storage/snapshots_grayscale/
    - dataset/archive/ (gambar referensi berkualitas tinggi)

Pipeline Segmentasi Karakter:
    1. Vertical Projection Profiling untuk menemukan gap antar karakter
    2. Connected Component Analysis (CCA) sebagai fallback
    3. Proportional Grid Slicing sebagai fallback terakhir
    4. Contour-based Refinement pada setiap potongan

Data Augmentation:
    - Rotasi kecil (±5°)
    - Perubahan brightness & contrast
    - Gaussian blur & noise
    - Erosi & dilasi morfologi
    - Elastic distortion ringan
    - Perspective warp ringan

Output:
    dataset/archive/DatasetCharacter/<KARAKTER>/
"""

import os
import re
import cv2
import random
import numpy as np
from collections import defaultdict

# ============================ KONFIGURASI ============================
SNAPSHOT_DIRS = [
    os.path.join("storage", "snapshots"),
    os.path.join("storage", "snapshots_grayscale"),
    os.path.join("dataset", "archive", "dataset"),
    os.path.join("dataset", "archive"),
]
TARGET_DATASET_DIR = os.path.join("dataset", "archive", "DatasetCharacter")
IMG_SIZE = (32, 32)
AUGMENT_PER_CHAR = 5           # Jumlah gambar augmentasi per karakter yang berhasil dipotong
MIN_CHAR_HEIGHT_RATIO = 0.25   # Minimum tinggi karakter relatif terhadap tinggi crop
MIN_CHAR_WIDTH_PX = 3          # Minimum lebar karakter (piksel)
MAX_CHAR_WIDTH_RATIO = 0.5     # Maksimum lebar karakter relatif terhadap lebar crop
RANDOM_SEED = 42
# =====================================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def preprocess_plate_image(img):
    """Pra-pemrosesan citra plat nomor: crop bawah, grayscale, auto-invert, CLAHE."""
    h, w = img.shape[:2]
    
    # Potong 18% bagian bawah (angka pajak)
    img_top = img[0:int(h * 0.82), :] if h > 18 else img
    crop_h, crop_w = img_top.shape[:2]

    # Grayscale
    if len(img_top.shape) == 3:
        gray = cv2.cvtColor(img_top, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_top.copy()

    # Auto-inversi jika plat hitam (latar gelap)
    if np.mean(gray) < 125:
        gray = cv2.bitwise_not(gray)

    # CLAHE untuk meningkatkan kontras
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    return enhanced, crop_h, crop_w


def vertical_projection_segment(binary_img, n_chars, crop_h, crop_w):
    """Segmentasi karakter menggunakan Vertical Projection Profiling."""
    # Hitung histogram vertikal (jumlah piksel putih per kolom)
    v_proj = np.sum(binary_img > 0, axis=0)
    
    # Normalisasi dan cari gap (kolom dengan sedikit piksel)
    threshold = crop_h * 0.15  # Threshold untuk gap
    
    # Cari segmen karakter (run-length encoding)
    in_char = False
    segments = []
    start = 0
    
    for x in range(crop_w):
        if v_proj[x] > threshold and not in_char:
            in_char = True
            start = x
        elif v_proj[x] <= threshold and in_char:
            in_char = False
            if x - start >= MIN_CHAR_WIDTH_PX:
                segments.append((start, x))
    
    # Tangani karakter terakhir yang masih berjalan
    if in_char and crop_w - start >= MIN_CHAR_WIDTH_PX:
        segments.append((start, crop_w))
    
    # Filter segmen yang terlalu lebar (mungkin 2 karakter bergabung)
    refined_segments = []
    for (sx, ex) in segments:
        seg_w = ex - sx
        if seg_w > crop_w * MAX_CHAR_WIDTH_RATIO and n_chars > 0:
            # Split menjadi 2
            mid = sx + seg_w // 2
            refined_segments.append((sx, mid))
            refined_segments.append((mid, ex))
        else:
            refined_segments.append((sx, ex))
    
    return refined_segments


def connected_component_segment(binary_img, crop_h, crop_w):
    """Segmentasi karakter menggunakan Connected Component Analysis."""
    # Inversi untuk connectedComponents (memerlukan foreground putih)
    inv = cv2.bitwise_not(binary_img)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
    
    segments = []
    for i in range(1, num_labels):  # Skip background (label 0)
        x, y, w, h, area = stats[i]
        
        # Filter berdasarkan ukuran
        if (h >= crop_h * MIN_CHAR_HEIGHT_RATIO and 
            w >= MIN_CHAR_WIDTH_PX and
            w <= crop_w * MAX_CHAR_WIDTH_RATIO and
            area >= 20):  # Minimum area
            segments.append((x, x + w))
    
    # Sort berdasarkan posisi x (kiri ke kanan)
    segments.sort(key=lambda s: s[0])
    
    # Merge segmen yang sangat berdekatan (mungkin bagian dari karakter yang sama)
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] < 3:  # Gap < 3 piksel
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)
    
    return merged


def proportional_grid_segment(n_chars, crop_w):
    """Fallback: Segmentasi karakter menggunakan grid proporsional sederhana."""
    if n_chars <= 0:
        return []
    
    char_step = crop_w / float(n_chars)
    segments = []
    for i in range(n_chars):
        x1 = int(i * char_step)
        x2 = int((i + 1) * char_step)
        segments.append((x1, x2))
    return segments


def refine_char_crop(char_region, crop_h):
    """Refine bounding box karakter menggunakan contour detection."""
    if char_region is None or char_region.size == 0:
        return char_region
    
    _, thresh = cv2.threshold(char_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(c)
        if rh >= int(crop_h * 0.25) and rw >= 2:
            refined = char_region[ry:ry+rh, rx:rx+rw]
            if refined.size > 0:
                return refined
    
    return char_region


def augment_image(img):
    """Menghasilkan satu gambar augmentasi dari gambar karakter."""
    result = img.copy()
    h, w = result.shape[:2]
    
    # 1. Rotasi kecil (±5°)
    if random.random() < 0.5:
        angle = random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # 2. Perubahan brightness & contrast
    if random.random() < 0.6:
        alpha = random.uniform(0.7, 1.3)  # Contrast
        beta = random.randint(-30, 30)     # Brightness
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
    
    # 3. Gaussian blur ringan
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)
    
    # 4. Gaussian noise
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(5, 20), result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 5. Erosi atau dilasi morfologi
    if random.random() < 0.3:
        kernel_size = random.choice([2, 3])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        if random.random() < 0.5:
            result = cv2.erode(result, kernel, iterations=1)
        else:
            result = cv2.dilate(result, kernel, iterations=1)
    
    # 6. Perspective warp ringan
    if random.random() < 0.3:
        margin = max(1, int(min(w, h) * 0.08))
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), random.randint(0, margin)],
            [random.randint(0, margin), h - random.randint(0, margin)],
            [w - random.randint(0, margin), h - random.randint(0, margin)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # 7. Salt & pepper noise
    if random.random() < 0.2:
        num_salt = int(result.size * 0.005)
        for _ in range(num_salt):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            result[y, x] = 255 if random.random() < 0.5 else 0
    
    return result


def segment_and_save_characters(image_path):
    """Memotong karakter dari gambar crop plat menggunakan multi-method segmentation pipeline."""
    filename = os.path.basename(image_path)
    
    # Ekstrak teks plat nomor dari nama file
    match = re.search(r"(?:_|^)([A-Z0-9]{4,10})\.jpg$", filename, re.IGNORECASE)
    if not match:
        return 0, {}

    plate_text = match.group(1).upper()
    if plate_text == "UNREAD" or len(plate_text) < 4:
        return 0, {}

    img = cv2.imread(image_path)
    if img is None:
        return 0, {}

    h, w = img.shape[:2]
    if h < 12 or w < 20:
        return 0, {}

    # Pra-pemrosesan
    enhanced, crop_h, crop_w = preprocess_plate_image(img)
    
    # Binarisasi untuk segmentasi
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    char_list = [c for c in list(plate_text) if c.isalnum()]
    n_chars = len(char_list)
    if n_chars < 4:
        return 0, {}

    # ---- Multi-Method Segmentation Pipeline ----
    # Method 1: Vertical Projection Profiling
    segments = vertical_projection_segment(binary, n_chars, crop_h, crop_w)
    
    # Method 2: Connected Component Analysis (jika projection gagal/mismatch)
    if len(segments) < n_chars * 0.6 or len(segments) > n_chars * 1.5:
        cca_segments = connected_component_segment(binary, crop_h, crop_w)
        if abs(len(cca_segments) - n_chars) < abs(len(segments) - n_chars):
            segments = cca_segments
    
    # Method 3: Proportional Grid Slicing (fallback terakhir)
    if len(segments) != n_chars:
        segments = proportional_grid_segment(n_chars, crop_w)

    # Pastikan jumlah segmen sama dengan jumlah karakter
    if len(segments) != n_chars:
        # Jika masih tidak cocok, paksa grid slicing
        segments = proportional_grid_segment(n_chars, crop_w)

    saved_count = 0
    char_counts = defaultdict(int)

    for i, char in enumerate(char_list):
        if i >= len(segments):
            break
            
        x1, x2 = segments[i]
        x1 = max(0, x1)
        x2 = min(crop_w, x2)
        
        char_crop = enhanced[:, x1:x2]
        if char_crop.size == 0 or char_crop.shape[0] < 5 or char_crop.shape[1] < 2:
            continue

        # Refine bounding box
        char_crop = refine_char_crop(char_crop, crop_h)
        if char_crop is None or char_crop.size == 0:
            continue

        if not (len(char) == 1 and char.isalnum()):
            continue

        # Resize ke ukuran standar
        char_resized = cv2.resize(char_crop, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        
        # Simpan gambar asli
        char_dir = os.path.join(TARGET_DATASET_DIR, char.upper())
        os.makedirs(char_dir, exist_ok=True)

        base_name = filename[:-4]
        out_filename = f"cctv_{base_name}_idx{i}.jpg"
        out_path = os.path.join(char_dir, out_filename)
        
        cv2.imwrite(out_path, char_resized)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            saved_count += 1
            char_counts[char] += 1

            # ---- Data Augmentation: Buat N variasi tambahan ----
            for aug_idx in range(AUGMENT_PER_CHAR):
                aug_img = augment_image(char_resized)
                aug_filename = f"aug{aug_idx}_{base_name}_idx{i}.jpg"
                aug_path = os.path.join(char_dir, aug_filename)
                cv2.imwrite(aug_path, aug_img)
                if os.path.exists(aug_path) and os.path.getsize(aug_path) > 0:
                    saved_count += 1
                    char_counts[char] += 1
        else:
            if os.path.exists(out_path):
                os.remove(out_path)

    return saved_count, char_counts


def print_dataset_statistics():
    """Menampilkan statistik distribusi dataset per kelas karakter."""
    if not os.path.exists(TARGET_DATASET_DIR):
        return
    
    print("\n" + "=" * 65)
    print("  STATISTIK DISTRIBUSI DATASET KARAKTER")
    print("=" * 65)
    
    total = 0
    stats = {}
    for cls_dir in sorted(os.listdir(TARGET_DATASET_DIR)):
        cls_path = os.path.join(TARGET_DATASET_DIR, cls_dir)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.endswith(".jpg")])
            stats[cls_dir] = count
            total += count
    
    if not stats:
        print("  [KOSONG] Belum ada data di dataset.")
        return
    
    min_count = min(stats.values())
    max_count = max(stats.values())
    avg_count = total / len(stats) if stats else 0
    
    # Print dalam format tabel
    print(f"  {'Karakter':<10} {'Jumlah':>8}  {'Bar'}")
    print(f"  {'-'*10} {'-'*8}  {'-'*30}")
    
    for char in sorted(stats.keys()):
        count = stats[char]
        bar_len = int((count / max_count) * 30) if max_count > 0 else 0
        bar = "#" * bar_len
        marker = " [!] KURANG" if count < avg_count * 0.5 else ""
        print(f"  {char:<10} {count:>8}  {bar}{marker}")
    
    print(f"\n  Total Sampel    : {total}")
    print(f"  Jumlah Kelas    : {len(stats)}")
    print(f"  Min per Kelas   : {min_count}")
    print(f"  Max per Kelas   : {max_count}")
    print(f"  Rata-rata       : {avg_count:.0f}")
    
    # Peringatan untuk kelas yang kurang representasi
    underrepresented = [c for c, v in stats.items() if v < avg_count * 0.5]
    if underrepresented:
        print(f"\n  [!] Kelas kurang representasi ({len(underrepresented)}): {', '.join(underrepresented)}")
        print("      Pertimbangkan menambah gambar referensi untuk karakter ini.")
    
    print("=" * 65)


def main():
    print("=" * 65)
    print("  ADVANCED DATASET BUILDER - Plat Nomor Indonesia")
    print("  Pipeline: Projection + CCA + Grid + Augmentation")
    print("=" * 65)
    
    os.makedirs(TARGET_DATASET_DIR, exist_ok=True)

    total_images = 0
    total_chars_extracted = 0
    global_char_counts = defaultdict(int)

    for s_dir in SNAPSHOT_DIRS:
        if not os.path.exists(s_dir):
            print(f"[SKIP] Direktori tidak ditemukan: '{s_dir}'")
            continue

        files = [os.path.join(s_dir, f) for f in os.listdir(s_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"\n[INFO] Memproses {len(files)} file dari '{s_dir}'...")

        for idx, f_path in enumerate(files):
            count, char_counts = segment_and_save_characters(f_path)
            if count > 0:
                total_images += 1
                total_chars_extracted += count
                for char, cnt in char_counts.items():
                    global_char_counts[char] += cnt
            
            # Progress indicator setiap 50 file
            if (idx + 1) % 50 == 0:
                print(f"  ... {idx + 1}/{len(files)} file diproses ({total_chars_extracted} karakter)")

    print("\n" + "=" * 65)
    print("  HASIL KONVERSI")
    print("=" * 65)
    print(f"  Snapshot diproses        : {total_images}")
    print(f"  Karakter diekstrak       : {total_chars_extracted}")
    print(f"  (termasuk augmentasi {AUGMENT_PER_CHAR}x per karakter asli)")
    print(f"  Output folder            : '{TARGET_DATASET_DIR}'")
    
    # Tampilkan statistik lengkap
    print_dataset_statistics()


if __name__ == "__main__":
    main()
