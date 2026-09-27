"""
Training Model Klasifikasi Karakter Plat Nomor Indonesia (0-9, A-Z) - Advanced
================================================================================
Script ini melatih model CNN yang lebih kuat (PlateCharNetV2) untuk mengenali
36 karakter plat nomor Indonesia.

Peningkatan dari versi sebelumnya:
    - Arsitektur CNN lebih dalam dengan Residual Connections
    - Training augmentation (RandomAffine, ColorJitter, GaussianBlur)
    - Learning Rate Scheduler (CosineAnnealing)
    - Label Smoothing pada Cross Entropy Loss
    - Early Stopping untuk mencegah overfitting
    - Class-weighted loss untuk mengatasi data imbalance
    - Evaluasi per-kelas (Confusion Matrix + Per-Class Accuracy)

Output:
    - models/char_model.pth (Model PyTorch terbaik)
    - models/char_labels.json (Mapping indeks kelas -> karakter)
    - models/training_report.json (Laporan metrik training)
"""

import os
import json
import zipfile
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, datasets

# ============================ KONFIGURASI ============================
ZIP_PATH = "archive.zip"
POSSIBLE_DATASET_DIRS = [
    os.path.join("dataset", "archive", "DatasetCharacter"),
    os.path.join("dataset", "archive"),
    "DatasetCharacter",
    "dataset_chars"
]
MODEL_SAVE_PATH = os.path.join("models", "char_model.pth")
LABEL_SAVE_PATH = os.path.join("models", "char_labels.json")
REPORT_SAVE_PATH = os.path.join("models", "training_report.json")
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
IMG_SIZE = (32, 32)
LABEL_SMOOTHING = 0.1
PATIENCE = 7       # Early stopping patience
MIN_DELTA = 0.001  # Minimum improvement untuk early stopping
# =====================================================================


class ResidualBlock(nn.Module):
    """Blok residual sederhana untuk memperdalam network tanpa vanishing gradient."""

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
    """
    Model CNN yang lebih kuat dengan Residual Connections untuk klasifikasi karakter plat nomor.
    
    Arsitektur:
        Conv(1→32) → ResBlock(32) → Pool → Conv(32→64) → ResBlock(64) → Pool →
        Conv(64→128) → ResBlock(128) → Pool → Conv(128→256) → GAP → FC(256→128) → FC(128→N)
    """

    def __init__(self, num_classes=36):
        super(PlateCharNetV2, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 → 16x16
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2, 2),

            # Block 2: 16x16 → 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.MaxPool2d(2, 2),

            # Block 3: 8x8 → 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.MaxPool2d(2, 2),

            # Block 4: 4x4 → Global Average Pooling
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Global Average Pooling menggantikan flatten
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Backward compatibility: alias nama lama agar main.py bisa memuat model baru
PlateCharNet = PlateCharNetV2


def get_dataset_dir():
    """Mendapatkan direktori 'DatasetCharacter' dari folder dataset/archive."""
    for path in POSSIBLE_DATASET_DIRS:
        if os.path.exists(path):
            # Cari direktori yang berisi folder kelas (0-9, A-Z)
            for root, dirs, files in os.walk(path):
                if len(dirs) >= 10:
                    return root

    if os.path.exists(ZIP_PATH):
        print(f"[INFO] Mengekstrak {ZIP_PATH}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall("dataset_chars")
        for root, dirs, files in os.walk("dataset_chars"):
            if len(dirs) >= 10:
                return root

    raise FileNotFoundError("[ERROR] Tidak dapat menemukan folder dataset karakter di 'dataset/archive/DatasetCharacter'")


def compute_class_weights(dataset):
    """Menghitung bobot kelas untuk mengatasi data imbalance."""
    targets = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(targets)
    total = len(targets)
    num_classes = len(class_counts)
    
    # Inverse frequency weighting
    weights = {}
    for cls, count in class_counts.items():
        weights[cls] = total / (num_classes * count)
    
    # Sample weights untuk WeightedRandomSampler
    sample_weights = [weights[t] for t in targets]
    
    # Class weights tensor untuk loss function
    class_weight_tensor = torch.zeros(num_classes)
    for cls, w in weights.items():
        class_weight_tensor[cls] = w
    
    return sample_weights, class_weight_tensor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Menggunakan Device: {device}")
    os.makedirs("models", exist_ok=True)

    # 1. Siapkan Dataset
    data_dir = get_dataset_dir()
    print(f"[INFO] Direktori Dataset Karakter: {data_dir}")

    # ---- Training Augmentation ----
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.RandomAffine(
            degrees=8,              # Rotasi ±8°
            translate=(0.08, 0.08), # Translasi 8%
            scale=(0.85, 1.15),     # Skala 85%-115%
            shear=5                 # Shear ±5°
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
        ], p=0.3),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3)
        ], p=0.4),
        transforms.RandomInvert(p=0.05),  # Sesekali inversi untuk robustness
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),  # Random occlusion kecil
    ])

    # ---- Validation Transform (tanpa augmentasi) ----
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset dua kali: sekali untuk train (dengan augmentasi), sekali untuk val
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(root=data_dir, transform=val_transform)
    
    num_classes = len(full_dataset.classes)
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"[INFO] Jumlah Sampel Gambar : {len(full_dataset)}")
    print(f"[INFO] Jumlah Kelas Karakter: {num_classes} ({full_dataset.classes})")

    # Simpan mapping kelas
    with open(LABEL_SAVE_PATH, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"[INFO] Mapping label tersimpan di: {LABEL_SAVE_PATH}")

    # Split Train (80%) dan Validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Gunakan generator fixed untuk reproducibility
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size], generator=generator)
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices.indices)

    # Hitung class weights dari training set
    print("[INFO] Menghitung bobot kelas untuk balancing...")
    train_targets = [full_dataset[i][1] for i in train_indices.indices]
    class_counts = Counter(train_targets)
    total_train = len(train_targets)
    
    sample_weights = [total_train / (num_classes * class_counts[t]) for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    class_weight_tensor = torch.zeros(num_classes)
    for cls, count in class_counts.items():
        class_weight_tensor[cls] = total_train / (num_classes * count)
    class_weight_tensor = class_weight_tensor.to(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Inisialisasi Model
    model = PlateCharNetV2(num_classes=num_classes).to(device)
    
    # Hitung jumlah parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 3. Training Loop dengan Early Stopping
    print(f"\n{'='*70}")
    print(f"  TRAINING MODEL PlateCharNetV2")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"  Label Smoothing: {LABEL_SMOOTHING} | Early Stopping: {PATIENCE} epochs")
    print(f"{'='*70}\n")
    
    best_val_acc = 0.0
    patience_counter = 0
    training_history = []

    for epoch in range(1, EPOCHS + 1):
        # ---- Training Phase ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping untuk stabilitas
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # ---- Validation Phase ----
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        # Per-class accuracy tracking
        class_correct = {i: 0 for i in range(num_classes)}
        class_total = {i: 0 for i in range(num_classes)}
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)
                
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = preds[i].item()
                    class_total[label] += 1
                    if pred == label:
                        class_correct[label] += 1

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_loss_avg = val_loss / val_total if val_total > 0 else 0
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Log epoch
        epoch_data = {
            "epoch": epoch,
            "train_loss": round(epoch_loss, 4),
            "train_acc": round(epoch_acc * 100, 2),
            "val_loss": round(val_loss_avg, 4),
            "val_acc": round(val_acc * 100, 2),
            "lr": round(current_lr, 6)
        }
        training_history.append(epoch_data)
        
        # Status indicator
        improved = ""
        if val_acc > best_val_acc + MIN_DELTA:
            improved = " [BEST]"
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'idx_to_class': idx_to_class,
                'val_acc': val_acc,
                'epoch': epoch
            }, MODEL_SAVE_PATH)
        else:
            patience_counter += 1
        
        print(
            f"  Epoch [{epoch:02d}/{EPOCHS:02d}] "
            f"Loss: {epoch_loss:.4f} | Train: {epoch_acc*100:.1f}% | "
            f"Val: {val_acc*100:.1f}% | ValLoss: {val_loss_avg:.4f} | "
            f"LR: {current_lr:.6f}{improved}"
        )
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n  [STOP] Early Stopping! Tidak ada peningkatan selama {PATIENCE} epoch.")
            break

    # 4. Evaluasi Final per-Kelas
    print(f"\n{'='*70}")
    print(f"  EVALUASI PER-KELAS (Model Terbaik)")
    print(f"{'='*70}")
    
    # Load model terbaik untuk evaluasi final
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = preds[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
    
    per_class_acc = {}
    print(f"  {'Karakter':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    
    problem_chars = []
    for cls_idx in sorted(class_correct.keys()):
        if class_total[cls_idx] > 0:
            acc = class_correct[cls_idx] / class_total[cls_idx]
            char_name = idx_to_class.get(str(cls_idx), idx_to_class.get(cls_idx, f"#{cls_idx}"))
            per_class_acc[char_name] = round(acc * 100, 1)
            marker = " [!]" if acc < 0.7 else ""
            print(f"  {char_name:<10} {class_correct[cls_idx]:>8} {class_total[cls_idx]:>8} {acc*100:>9.1f}%{marker}")
            if acc < 0.7:
                problem_chars.append(char_name)
    
    if problem_chars:
        print(f"\n  [!] Karakter bermasalah (akurasi < 70%): {', '.join(problem_chars)}")
        print("      Tips: Tambahkan lebih banyak sampel atau periksa kualitas label.")
    
    # 5. Simpan Report
    report = {
        "model": "PlateCharNetV2",
        "best_val_accuracy": round(best_val_acc * 100, 2),
        "num_classes": num_classes,
        "total_params": total_params,
        "training_epochs": len(training_history),
        "per_class_accuracy": per_class_acc,
        "problem_characters": problem_chars,
        "history": training_history
    }
    
    with open(REPORT_SAVE_PATH, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  TRAINING SELESAI!")
    print(f"  Model terbaik  : '{MODEL_SAVE_PATH}' (Val Acc: {best_val_acc*100:.2f}%)")
    print(f"  Label mapping  : '{LABEL_SAVE_PATH}'")
    print(f"  Training report: '{REPORT_SAVE_PATH}'")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
