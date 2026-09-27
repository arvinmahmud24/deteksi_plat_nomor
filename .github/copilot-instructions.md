# PLAT - Indonesian ANPR System (Copilot Instructions)

## System Architecture Overview

**PLAT** is a real-time Indonesian License Plate Recognition (ANPR) system that monitors CCTV streams 24/7 and logs detections to daily Excel files with embedded plate images.

### Core Pipeline
```
CCTV Stream → YOLOv8 (detect plates) → ByteTrack (vehicle tracking) 
→ AsyncOCRWorker (parallel OCR) → Multi-Frame Voting → Daily Excel Log + Photo snapshots
```

### Data Flow
1. **Frame Capture**: YOLOv8 detects plate regions in real-time (DET_CONF=0.35)
2. **Tracking**: ByteTrack maintains vehicle IDs across frames (MAX_TRACK_AGE_FRAMES=100)
3. **Async OCR Queue**: Parallel worker pool processes crops (background thread, zero-lag UI)
4. **Confidence Voting**: Aggregates multi-frame OCR results per vehicle ID, picks highest confidence reading
5. **Validation**: Strict Indonesian format: `[1-2 letters] [1-4 digits] [0-3 letters]` (see VALID_REGION_CODES)
6. **Logging**: Saves to `data/log_plat_YYYY-MM-DD.xlsx` with embedded grayscale crop image in Column F

### Key Components

| File | Purpose |
|------|---------|
| [main.py](../main.py) | Core ANPR pipeline (YOLOv8, ByteTrack, EasyOCR, Excel logging) |
| [train_character.py](../train_character.py) | PyTorch CNN training for character classification (36 classes: 0-9, A-Z) |
| [prepare_snapshot_dataset.py](../prepare_snapshot_dataset.py) | Auto character segmentation from CCTV snapshots for continuous learning |

## Critical Patterns & Implementation Details

### 1. Multi-Frame Weighted Confidence Voting
**Location**: [main.py](../main.py#L500-L600) - `finalize_readings()` function

Instead of logging the first OCR result, the system accumulates readings per vehicle ID:
- Each frame's OCR result (plate + confidence) stored in `readings[track_id]`
- Voting triggers when: vehicle lost (`MAX_TRACK_AGE_FRAMES`) OR sufficient frames collected (`BUFFER_FRAMES_AFTER_CROSS`)
- **Decision**: Pick plate text with highest cumulative confidence
- **Why**: Single-frame OCR is error-prone; multiple frames give robust consensus

**Key config values:**
```python
MIN_READINGS_TO_LOG = 1        # Log even 1-frame detections (set to 1 for gate security)
MIN_OCR_CONF = 0.55            # Minimum per-frame confidence threshold
DEBOUNCE_SECONDS = 20          # Prevent duplicate logging of same vehicle
BUFFER_FRAMES_AFTER_CROSS = 10 # Collect this many frames before voting
```

### 2. Async OCR Worker Pool
**Location**: [main.py](../main.py#L330-L415) - `AsyncOCRWorker` class

Non-blocking multithreaded worker that processes OCR in background:
- Queue-based task dispatch (`self.task_queue.put((tid, crop))`)
- Daemon thread runs `_worker_loop()` continuously
- **Hybrid OCR**: CNN (if trained) → EasyOCR (always as fallback)
- **Why**: Prevents frame-processing lag; maintains high FPS on GUI preview

**Usage pattern:**
```python
ocr_worker.enqueue(track_id, plate_crop)  # Non-blocking, drops if queue full
# Results automatically update readings[] and best_crops[] dicts (thread-safe via lock)
```

### 3. Hybrid CNN + EasyOCR Recognition
**Location**: [main.py](../main.py#L410-L445)

Two-method approach with confidence merging:
- **Method 1 (CNN)**: If `models/char_model.pth` exists, uses custom PlateCharNetV2 for character classification
- **Method 2 (EasyOCR)**: Always runs as fallback (handles unseen formats better)
- **Merge logic**: 
  - If only one succeeds → use that result
  - If both succeed → pick highest confidence
  - If **both agree** → boost confidence by +5% (high trust signal)

**Important**: EasyOCR works best with **color upscaled images** (not binarized). See `preprocess_crop()`: sends 64px+ height color crops to EasyOCR, not binary images.

### 4. Indonesian License Plate Format Validation
**Location**: [main.py](../main.py#L250-L310) - `clean_plate()` function

Strict format parser with region code lookup:
```
[1-2 region letters] [1-4 digits] [optional 0-3 suffix letters]
Example: "B 1234 AB" (Jakarta region B, vehicle 1234, suffix AB)
```

- **VALID_REGION_CODES**: hardcoded dict maps 1-2 letter codes to provinces (B→Jakarta, D→Bandung, etc.)
- **Fallback parsing**: If space-separated fails, tries regex extraction and proportional slicing
- **Rejects**: Invalid regions, non-numeric middle section, malformed suffixes

**Why this matters**: Raw OCR often outputs "B1234AB" or "B 1234 AB" with inconsistent spacing. `clean_plate()` canonicalizes this.

### 5. Daily Excel Log with Embedded Images
**Location**: [main.py](../main.py#L520-L600) - `save_to_excel()` function

Automatic daily rollover to new Excel file:
- File path: `data/log_plat_YYYY-MM-DD.xlsx`
- **Embedding**: Grayscale crop image auto-embedded in Column F (cell height ~45px for readability)
- **Fallback**: If Excel locked by user → saves to `data/log_plat_backup_YYYY-MM-DD.csv`
- **Columns**: No, Timestamp, Plate, Confidence, Camera Name, [Image], Color Path, Grayscale Path

**Pattern**:
```python
save_to_excel(timestamp, plate, conf, camera, color_path, gray_path)
# Creates/opens data/log_plat_YYYY-MM-DD.xlsx, appends row, embeds image, saves
```

### 6. Character Dataset Auto-Generator
**Location**: [prepare_snapshot_dataset.py](../prepare_snapshot_dataset.py)

Pipeline for continuous learning from production snapshots:
1. **Character Segmentation**: Vertical projection profiling (primary) → Connected Component Analysis (fallback) → Grid slicing (last resort)
2. **Validation**: Characters must be 25-50% of plate height, width-constrained
3. **Augmentation**: Rotation ±5°, brightness/contrast jitter, blur, morph erosion/dilation, elastic distortion
4. **Output**: `dataset/archive/DatasetCharacter/[0-9A-Z]/` folders with augmented char images

**Usage**:
```bash
python prepare_snapshot_dataset.py  # Reads storage/snapshots/ + snapshots_grayscale/
python train_character.py            # Trains PlateCharNetV2 on DatasetCharacter/, outputs models/char_model.pth
```

### 7. PlateCharNetV2 Architecture
**Location**: [train_character.py](../train_character.py#L60-L95), [main.py](../main.py#L90-L120)

Custom CNN with residual connections:
- **Input**: 32×32 grayscale character images
- **Architecture**: Conv(1→32) → ResBlock(32) → MaxPool → Conv(32→64) → ResBlock(64) → MaxPool → Conv(64→128) → ResBlock(128) → MaxPool → Conv(128→256) → GlobalAvgPool → FC(256→128) → FC(128→36)
- **Training features**: Label smoothing, class-weighted loss, early stopping (patience=7), cosine learning rate schedule
- **Why ResBlocks**: Prevents vanishing gradients in deeper networks; improves convergence on small (32×32) character images

## Developer Workflows

### Running the System
```bash
# Quick start (uses default CCTV URL from code)
python main.py

# Custom video source
python main.py --source 0 --name "Webcam"                    # Webcam
python main.py --source "video.mp4" --name "Test"            # Video file
python main.py --source "rtsp://admin:pass@192.168.1.10:554" --name "Gate1"  # IP Camera
```

### Building/Improving the Character Model
```bash
# 1. Collect production snapshots (automatic during main.py runs)
# 2. Auto-segment characters from snapshots
python prepare_snapshot_dataset.py

# 3. Inspect generated dataset
ls dataset/archive/DatasetCharacter/  # Should see 0-9, A-Z subdirs with ~500-1000 images each

# 4. Train model
python train_character.py
# Outputs: models/char_model.pth, models/char_labels.json, models/training_report.json

# 5. Deploy
python main.py  # Will auto-load new char_model.pth if exists
```

### Debugging Production Issues
- **Low OCR accuracy**: Check `models/training_report.json` per-class accuracy; dataset may be biased (e.g., poor region code coverage)
- **Frame drops/lag**: Increase `AsyncOCRWorker` queue size or reduce `OCR_EVERY_N_FRAMES`
- **Excel lock conflicts**: Check `data/log_plat_backup_*.csv` — data is safe there while Excel is open
- **YOLOv8 misses plates**: Reduce `DET_CONF` threshold (currently 0.35); may increase false positives

## Configuration & Tuning

**[main.py](../main.py#L50-L80)** contains all critical parameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `DEBOUNCE_SECONDS` | 20 | Prevent duplicate logging of same vehicle within 20s |
| `DET_CONF` | 0.35 | YOLOv8 detection confidence (lower = more detections, more false positives) |
| `MIN_OCR_CONF` | 0.55 | Minimum confidence per OCR frame (raised from 0.4 to reduce noise) |
| `MIN_PLATE_WIDTH` | 40 | Reject crops narrower than 40px (too small for reliable OCR) |
| `MAX_TRACK_AGE_FRAMES` | 100 | Forget vehicle after 100 frames without detection |
| `BUFFER_FRAMES_AFTER_CROSS` | 10 | Collect 10 OCR results before voting (higher = more stable, slower) |
| `OCR_EVERY_N_FRAMES` | 2 | Process OCR every 2nd frame per vehicle (balance: CPU vs accuracy) |

**Performance tuning**:
- **CPU-bound**: Increase `OCR_EVERY_N_FRAMES` to 3-4, or reduce `DET_CONF` for fewer detections
- **Accuracy-bound**: Decrease `MIN_OCR_CONF` to 0.40, increase `BUFFER_FRAMES_AFTER_CROSS` to 15

## Directory Structure & File Roles

```
models/
  ├── plate.pt                  # YOLOv8 weights (detects Indonesian plates)
  ├── char_model.pth            # PyTorch CNN (classifies individual characters)
  ├── char_labels.json          # {0: '0', 1: '1', ..., 35: 'Z'} mapping
  └── training_report.json      # Per-class accuracy + training metrics

storage/
  ├── snapshots/                # Original color crops (BGR)
  └── snapshots_grayscale/      # Preprocessed grayscale + CLAHE crops

data/
  ├── log_plat_YYYY-MM-DD.xlsx  # Daily Excel log with embedded images
  └── log_plat_backup_*.csv     # Fallback CSV if Excel locked

dataset/archive/
  └── DatasetCharacter/         # Training data for char_model.pth
      ├── 0/, 1/, ..., 9/       # Digit folders
      └── A/, B/, ..., Z/       # Letter folders
```

## External Dependencies & Integration

- **YOLOv8** (`ultralytics`): Plate detection. Models from Ultralytics hub or custom fine-tuned weights
- **ByteTrack**: Vehicle tracking across frames. No config needed; auto-integrated in YOLOv8
- **EasyOCR**: Text recognition. Auto-downloads language models on first run
- **PyTorch** (`torch`, `torchvision`): Character classification CNN training
- **OpenPyXL**: Excel file generation with image embedding
- **OpenCV** (`cv2`): Image I/O and preprocessing
- **Pillow** (`PIL`): Font rendering for overlay badges

**First-run behavior**: EasyOCR downloads ~50MB language models on first `reader.readtext()` call. Subsequent runs use cached models.

## Testing & Validation

- **Unit test**: `python train_character.py` includes internal train/val split (80/20) with per-class accuracy reporting
- **Integration test**: Run `python main.py --source "test_video.mp4"` to validate full pipeline on pre-recorded sample
- **Regression check**: Compare `models/training_report.json` metrics across model training runs to spot accuracy degradation

## Privacy & Compliance

System respects **UU No. 27 Tahun 2022** (Indonesia's Personal Data Protection Law):
- Automatic daily log rollover encourages retention policy enforcement
- File-level access control recommended (restrict `data/` and `storage/` folders to authorized personnel)
- Photos are cropped (only plate visible, no driver/passenger faces)
