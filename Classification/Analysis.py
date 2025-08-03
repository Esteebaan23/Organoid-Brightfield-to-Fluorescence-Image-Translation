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

        return img, label, img_path

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



base_path = "/home/UNT/hel0057/Documents/Organoids"  # ajusta si estás en entorno local
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_paths = load_image_paths(base_path)
train_data, val_data = split_data(image_paths)
train_tf, val_tf = get_transforms()
train_set = OrganoidDataset(train_data, transform=train_tf)
val_set = OrganoidDataset(val_data, transform=val_tf, apply_clahe=False)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

# Cargar modelo entrenado
model = HybridResNetViT().to(device)
model.load_state_dict(torch.load("best_model_CH4.pth", map_location=device))
model.eval()
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

results = []
y_true, y_pred = [], []

model.eval()
with torch.no_grad():
    for img, label, path in tqdm(val_loader):
        img = img.to(device)
        output = model(img)  # [1, 1] → salida sigmoidal
        prob = output.item()
        pred = int(prob > 0.5)
        true_label = int(label.item())

        y_true.append(true_label)
        y_pred.append(pred)

        results.append({
            "path": path[0],
            "pred": pred,
            "true": true_label,
            "prob": prob,
            "correct": int(pred == true_label)
        })

# Métricas
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Guardar resultados
df = pd.DataFrame(results)
df.to_excel("results.xlsx", index=False)
print("✅ Resultados guardados en results.xlsx")

import matplotlib.pyplot as plt
import cv2
import numpy as np

# Grad-CAM ResNet
def gradcam_resnet(model, img_tensor, device):
    features, gradients = [], []

    def fw_hook(m, i, o): features.append(o)
    def bw_hook(m, gi, go): gradients.append(go[0])

    handle_fw = model.resnet.layer4.register_forward_hook(fw_hook)
    handle_bw = model.resnet.layer4.register_backward_hook(bw_hook)

    output = model(img_tensor.to(device))
    output.backward(torch.ones_like(output))

    grads = gradients[0].detach().cpu()[0]
    feats = features[0].detach().cpu()[0]
    weights = torch.mean(grads, dim=(1, 2))
    cam = torch.zeros(feats.shape[1:])

    for i, w in enumerate(weights):
        cam += w * feats[i]
    cam = np.maximum(cam.numpy(), 0)
    cam = cam / np.max(cam)

    handle_fw.remove()
    handle_bw.remove()
    return cam

# Atención DeiT
import math

def vit_attention_map(model, img_tensor, device):
    import math

    attns = []

    def hook_fn(m, i, o): attns.append(o)
    handle = model.vit.blocks[-1].attn.register_forward_hook(hook_fn)

    _ = model(img_tensor.to(device))
    handle.remove()

    attn = attns[0]
    if attn.dim() == 4:
        attn_cls = attn[0].mean(0)[0, 1:]  # [CLS] → tokens
    elif attn.dim() == 3:
        attn_cls = attn.mean(0)[0, 1:]
    else:
        raise ValueError(f"Unexpected attention shape: {attn.shape}")

    num_patches = attn_cls.shape[0]

    # Buscar la mejor forma rectangular para reshaping
    best_h, best_w = None, None
    min_diff = float('inf')
    for h in range(1, int(math.sqrt(num_patches)) + 2):
        if num_patches % h == 0:
            w = num_patches // h
            if abs(h - w) < min_diff:
                best_h, best_w = h, w
                min_diff = abs(h - w)

    if best_h is None:
        raise ValueError(f"No rectangular shape found for {num_patches} tokens.")

    attn_map = attn_cls.reshape(best_h, best_w).detach().cpu().numpy()
    attn_map = cv2.resize(attn_map, (img_tensor.shape[2], img_tensor.shape[3]))
    attn_map = attn_map / np.max(attn_map)
    return attn_map


# Visualización
def plot_and_save(original, cam, attn_map, prob, pred, true_label, save_path):
    # Asegura mismo tamaño
    target_shape = attn_map.shape  # (H, W)

    cam_resized = cv2.resize(cam, (target_shape[1], target_shape[0]))
    combined = (cam_resized + attn_map) / 2

    original = np.array(original.resize((target_shape[1], target_shape[0])))
    heat_cam = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heat_attn = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    heat_comb = cv2.applyColorMap(np.uint8(255 * combined), cv2.COLORMAP_JET)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")

    axes[1].imshow(original)
    axes[1].imshow(cam_resized, cmap='jet', alpha=0.5)
    axes[1].set_title(f"Grad-CAM\nPred: {pred} ({prob:.2f}) | True: {true_label}")

    axes[2].imshow(original)
    axes[2].imshow(attn_map, cmap='jet', alpha=0.5)
    axes[2].set_title("Token Attention")

    axes[3].imshow(original)
    axes[3].imshow(combined, cmap='jet', alpha=0.5)
    axes[3].set_title("Combined")

    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Procesar errores
os.makedirs("visual_results", exist_ok=True)
df = pd.read_excel("results.xlsx")
df_errors = df[df["correct"] == 0]

for idx, row in tqdm(df_errors.iterrows(), total=len(df_errors)):
    img_path = row["path"]
    prob = row["prob"]
    pred = row["pred"]
    true_label = row["true"]

    original = Image.open(img_path).convert("RGB")
    img_tensor = val_tf(original).unsqueeze(0)

    cam = gradcam_resnet(model, img_tensor, device)
    attn_map = vit_attention_map(model, img_tensor, device)

    out_path = f"visual_results/sample_{idx}_pred{pred}_true{true_label}.png"
    plot_and_save(original, cam, attn_map, prob, pred, true_label, out_path)

print("✅ Visualizaciones exportadas en visual_results/")


