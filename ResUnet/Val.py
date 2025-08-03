import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import skimage.metrics
from Model.Model import ResUNetGenerator
from Utils.utils import FluorescenceDataset
from torch.utils.data import random_split


def compute_psnr(pred, target):
    pred = pred.clamp(0, 1).cpu().numpy()
    target = target.clamp(0, 1).cpu().numpy()
    return skimage.metrics.peak_signal_noise_ratio(target, pred, data_range=1.0)

def compute_ssim(pred, target):
    pred = pred.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    target = target.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return skimage.metrics.structural_similarity(target, pred, channel_axis=2, data_range=1.0)

# === Settings (One for model) ===
model_path = "Files/gen_epoch_104.pth" 
output_dir = "val_outputs_Resunet_NoForm_104"
os.makedirs(output_dir, exist_ok=True)
batch_size = 1  # Inference 1 per image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORMACIÓN Y CARGA DE DATOS ===
transform = transforms.Compose([
    transforms.Resize((720, 960)),
    transforms.ToTensor()
])


full_dataset = FluorescenceDataset(
    "/home/UNT/hel0057/Documents/Organoids/Chamber Forming/CH4",
    "/home/UNT/hel0057/Documents/Organoids/Chamber Forming/CH1",
    transform
)

# Train and Val Split 70% and 30%
train_size = int(0.7 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# === Load Generative model ===
gen = ResUNetGenerator().to(device)
gen.load_state_dict(torch.load(model_path, map_location=device))
gen.eval()

# === Inference and Metrics ===
results = []

with torch.no_grad():
    for i, (gray, color) in enumerate(tqdm(val_loader)):
        gray, color = gray.to(device), color.to(device)
        pred = gen(gray)
        gray_3c = gray.repeat(1, 3, 1, 1)
        psnr = compute_psnr(pred[0], color[0])
        ssim = compute_ssim(pred[0], color[0])
        comparison = torch.cat([gray_3c[0], color[0], pred[0]], dim=2)  # horizontal stack
        save_image(comparison, f"{output_dir}/val_{i+1:04d}.png")
        results.append({
            "Image": f"val_{i+1:04d}.png",
            "PSNR": psnr,
            "SSIM": ssim
        })

df = pd.DataFrame(results)
df.to_excel(os.path.join(output_dir, "val_metrics.xlsx"), index=False)
print("✅ Validation Done.")