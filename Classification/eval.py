import os
import glob
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import cv2
# Dataset
def load_image_paths(base_dir, channel='CH4'):
    classes = ['Chamber Forming', 'Chamber Nonforming']
    data = []
    for label in classes:
        img_dir = os.path.join(base_dir, label, channel)
        for img_path in glob.glob(f"{img_dir}/*.tif"):
            data.append((img_path, 0 if label == 'Chamber Forming' else 1))
    return data

def split_data(image_list, test_size=0.2):
    train_data, val_data = train_test_split(image_list, test_size=test_size, stratify=[label for _, label in image_list])
    return train_data, val_data

class OrganoidDataset(Dataset):
    def __init__(self, data, transform=None, apply_clahe=True):
        self.data = data
        self.transform = transform
        self.apply_clahe = apply_clahe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))

        if self.apply_clahe:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.transform:
            img = self.transform(img)

        return img, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    return train_transform, val_transform
class HybridResNetViT(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ResNet50
        self.resnet = models.resnet50(pretrained=True)
        self.resnet_out = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # DeiT3 ViT
        self.vit = timm.create_model('deit3_base_patch16_224', pretrained=True)
        self.vit_out = self.vit.head.in_features
        self.vit.head = nn.Identity()

        # Clasificación final (sigmoid para binaria)
        self.fc = nn.Sequential(
            nn.Linear(self.resnet_out + self.vit_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.resnet(x)  # [B, 2048]
        x2 = self.vit(x)     # [B, 768]
        x = torch.cat((x1, x2), dim=1)  # [B, 2816]
        return self.fc(x)


# Obtener rutas
def get_data(base_path):
    data = []
    for label_name, label_val in [('Chamber Forming', 1), ('Chamber Nonforming', 0)]:
        ch4_path = os.path.join(base_path, label_name, 'CH4')
        for f in glob.glob(os.path.join(ch4_path, '*.tif')):
            data.append((f, label_val))
    return data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def simple_val(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels, *_ in dataloader:
            imgs = imgs.to(device)

            outputs = model(imgs).squeeze().cpu().numpy()
            labels = labels.cpu().numpy()

            preds = (outputs > 0.5).astype(int)

            y_true.extend(np.atleast_1d(labels))
            y_pred.extend(np.atleast_1d(preds))

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    return acc, precision, recall, f1, (y_true, y_pred)


# Preparar datos
base_path = "/home/UNT/hel0057/Documents/Organoids"  # ajusta si estás en entorno local
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_paths = load_image_paths(base_path)
train_data, val_data = split_data(image_paths)
train_tf, val_tf = get_transforms()

train_set = OrganoidDataset(train_data, transform=train_tf)
val_set = OrganoidDataset(val_data, transform=val_tf, apply_clahe=False)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=2)

model = HybridResNetViT().to(device)
model.load_state_dict(torch.load("best_model_CH4.pth", map_location=device))
val_acc, precision, recall, f1, _ = simple_val(model, val_loader, device)
print(val_acc)
print(precision)
print(recall)
print(f1)