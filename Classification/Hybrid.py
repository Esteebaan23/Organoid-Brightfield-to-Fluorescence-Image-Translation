import os
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
import openpyxl

import glob
from sklearn.model_selection import train_test_split

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

import torch
import torch.nn as nn
import timm
from torchvision import models
import torch.nn.functional as F

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




def train_model(model, train_loader, val_loader, save_path, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    best_acc = 0
    best_f1 = 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs, labels = imgs.to(device), labels.float().to(device)  # float para BCELoss

            outputs = model(imgs).squeeze()  # salida tipo [B]
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

            preds = (outputs > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc, precision, recall, f1, _ = simple_val(model, val_loader, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": running_loss / total,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f} | F1 = {f1:.4f}")

        if val_acc > best_acc and f1 > best_f1:
            best_acc = val_acc
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model with val acc: {val_acc:.4f}")

        torch.cuda.empty_cache()

    # Exportar métricas a Excel
    df = pd.DataFrame(history)
    df.to_excel(save_path.replace('.pth', '_history.xlsx'), index=False)


     


from sklearn.metrics import precision_score, recall_score, f1_score

def simple_val(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.cpu().numpy()

            outputs = model(imgs).squeeze().cpu().numpy()
            preds = (outputs > 0.5).astype(int)

            y_true.extend(labels)
            y_pred.extend(preds)

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    return acc, precision, recall, f1, (y_true, y_pred)



def evaluate_with_gradcam(model, val_loader, save_prefix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    y_true, y_pred = [], []
    all_imgs = []
    misclassified_idxs = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            for i in range(imgs.size(0)):
                y_true.append(labels[i].item())
                y_pred.append(preds[i].item())
                all_imgs.append(imgs[i].cpu())
                if preds[i].item() != labels[i].item():
                    misclassified_idxs.append(len(all_imgs) - 1)

    # Reporte y matriz
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    df_metrics = pd.DataFrame(report).transpose()
    df_metrics['confusion_matrix_0_0'] = cm[0, 0]
    df_metrics['confusion_matrix_0_1'] = cm[0, 1]
    df_metrics['confusion_matrix_1_0'] = cm[1, 0]
    df_metrics['confusion_matrix_1_1'] = cm[1, 1]
    df_metrics.to_excel(f"{save_prefix}_metrics.xlsx")

    # Grad-CAM en errores
    for idx in misclassified_idxs[:10]:  # máx 10
        img_tensor = all_imgs[idx].unsqueeze(0).to(device)
        generate_gradcam(model, img_tensor, y_true[idx], y_pred[idx], f"{save_prefix}_gradcam_error_{idx}.png")

    print(f"✅ Final evaluation done. Metrics + Grad-CAM saved.")


def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    all_imgs, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
            all_imgs.extend(imgs.cpu())
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(pd.DataFrame(report).transpose())
    print("Confusion Matrix:\n", cm)

    # Grad-CAM para imágenes mal clasificadas
    incorrects = [i for i, (p, t) in enumerate(zip(all_preds, all_labels)) if p != t]
    for idx in incorrects[:5]:  # solo 5 ejemplos
        img_tensor = all_imgs[idx].unsqueeze(0).to(device)
        generate_gradcam(model, img_tensor, all_labels[idx], all_preds[idx], f"error_{idx}.png")

    return np.mean([report[str(i)]['recall'] for i in range(2)]), report

def generate_gradcam(model, img_tensor, label, pred, save_path):
    model.eval()
    
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Registra hooks
    handle_fw = model.resnet.layer4.register_forward_hook(forward_hook)
    handle_bw = model.resnet.layer4.register_backward_hook(backward_hook)

    output = model(img_tensor)
    class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    # Extraer activaciones y gradientes
    activations = features[0].detach().cpu()
    grads = gradients[0].detach().cpu()

    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[0, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    # Visualizar sobre la imagen original
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5 + 0.5) * 255
    img = np.uint8(img)

    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, 0.6, img, 0.4, 0)
    cv2.imwrite(save_path, superimposed_img)

    # Eliminar hooks
    handle_fw.remove()
    handle_bw.remove()


base_dir = "/home/UNT/hel0057/Documents/Organoids"

for channel in ['CH4', 'CH1']:
    print(f"\n=== Entrenando canal {channel} ===")
    image_paths = load_image_paths(base_dir, channel)
    train_data, val_data = split_data(image_paths)
    train_tf, val_tf = get_transforms()

    train_set = OrganoidDataset(train_data, transform=train_tf)
    val_set = OrganoidDataset(val_data, transform=val_tf, apply_clahe=False)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=2)

    model = HybridResNetViT()
    checkpoint_path = "best_model_CH4.pth"
    train_model(model, train_loader, val_loader, checkpoint_path, num_epochs=50)

    # Validación final + Grad-CAM
    #model.load_state_dict(torch.load(checkpoint_path))
    #evaluate_with_gradcam(model, val_loader, save_prefix="CH4_results")



