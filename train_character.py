"""
Training Model Klasifikasi Karakter Plat Nomor Indonesia (0-9, A-Z)
===================================================================
Script ini mengekstrak 'DatasetCharacter' dari 'archive.zip' dan melatih
model Neural Network ringan (PlateCharNet / MobileNet) untuk mengenali
36 karakter plat nomor Indonesia secara presisi & ultra-cepat di CPU.

Output:
    - char_model.pth (Model PyTorch terlatih)
    - char_labels.json (Mapping indeks kelas -> karakter)
"""

import os
import zipfile
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = (32, 32)
# =====================================================================


class PlateCharNet(nn.Module):
    """Model Convolutional Neural Network (CNN) ultra-ringan khusus karakter plat nomor."""

    def __init__(self, num_classes=36):
        super(PlateCharNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Menggunakan Device: {device}")

    # 1. Ekstrak & Siapkan Dataset
    data_dir = get_dataset_dir()
    print(f"[INFO] Direktori Dataset Karakter: {data_dir}")

    # Transformasi Gambar: Grayscale 32x32 + Normalisasi
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
    num_classes = len(full_dataset.classes)
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"[INFO] Jumlah Sampel Gambar : {len(full_dataset)}")
    print(f"[INFO] Jumlah Kelas Karakter: {num_classes} ({full_dataset.classes})")

    # Simpan mapping kelas ke JSON
    with open(LABEL_SAVE_PATH, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"[INFO] Mapping label tersimpan di: {LABEL_SAVE_PATH}")

    # Split Train (80%) dan Validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Inisialisasi Model & Loss Function
    model = PlateCharNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n[INFO] Memulai Training Model Klasifikasi Karakter...")
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
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
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct.double() / total

        # Evaluasi Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        val_acc = val_correct.double() / val_total
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] - Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\n[SELESAI] Model Karakter Terbaik tersimpan di '{MODEL_SAVE_PATH}' dengan Akurasi Val: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
