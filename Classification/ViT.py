import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import vit_b_16, ViT_B_16_Weights

import openpyxl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OrganoidDataset(Dataset):
    def __init__(self, image_paths, labels, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment

        self.transform = A.Compose([
            A.Resize(224, 224),  
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8)),  
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(),
            ToTensorV2()
        ]) if augment else A.Compose([
            A.Resize(224, 224),  # 🔧 Añade esto también aquí
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8)),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        image = self.transform(image=image)["image"]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

def load_image_paths(base_dir, channel="CH4"):
    class_map = {"Chamber Forming": 1, "Chamber Nonforming": 0}
    image_paths = []
    labels = []

    for cls in ["Chamber Forming", "Chamber Nonforming"]:
        cls_path = os.path.join(base_dir, cls, channel)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):

                image_paths.append(os.path.join(cls_path, fname))
                labels.append(class_map[cls])
    
    return image_paths, labels

base_path = "/home/UNT/hel0057/Documents/Organoids"


def get_dataloaders(image_paths, labels, batch_size=16):
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42)

    train_dataset = OrganoidDataset(X_train, y_train, augment=True)
    val_dataset = OrganoidDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader




import timm
import torch.nn as nn

def create_model(device):
    model = timm.create_model('deit3_base_patch16_224', pretrained=True)
    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, 1),
        nn.Sigmoid()
    )
    return model.to(device)

def train_model(model, train_loader, val_loader, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCELoss()
    best_f1 = 0
    best_acc = 0
    history = []
    for epoch in range(50):
        model.train()
        train_losses = []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        preds = []
        true = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds.extend(outputs.cpu().numpy())
                true.extend(labels.numpy())

        preds_binary = (np.array(preds) > 0.6).astype(int)
        acc = accuracy_score(true, preds_binary)
        prec = precision_score(true, preds_binary)
        rec = recall_score(true, preds_binary)
        f1 = f1_score(true, preds_binary)

        history.append([epoch, np.mean(train_losses), acc, prec, rec, f1])
        print(f"Epoch {epoch+1}: Train Loss={np.mean(train_losses):.4f} | Val Acc={acc:.4f} F1={f1:.4f}")

        if f1 > best_f1 and acc > best_acc:
            best_f1 = f1
            best_acc = acc
            print(f"Model saved in epoch {epoch+1}")
            torch.save(model.state_dict(), save_path)

    return history, preds_binary, true

def save_metrics(history, preds, true_labels, excel_path):
    df = pd.DataFrame(history, columns=["Epoch", "Train Loss", "Val Accuracy", "Precision", "Recall", "F1"])
    df.to_excel(excel_path, index=False)

    cm = confusion_matrix(true_labels, preds)
    print("Confusion Matrix:")
    print(cm)

for channel in ["CH4", "CH1"]:
    print(f"\n==== Channel {channel} ====")
    image_paths, labels = load_image_paths(base_path, channel=channel)
    train_loader, val_loader = get_dataloaders(image_paths, labels)
    model = create_model(device)

    checkpoint_path = f"ViT_{channel}_best.pth"
    excel_path = f"metrics_{channel}.xlsx"

    history, preds, true_labels = train_model(model, train_loader, val_loader, checkpoint_path)
    save_metrics(history, preds, true_labels, excel_path)
