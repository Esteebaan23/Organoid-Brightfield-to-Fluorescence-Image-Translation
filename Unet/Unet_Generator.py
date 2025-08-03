import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# --- CBAM Module ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import random

# CBAM: Convolutional Block Attention Module
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid_channel = nn.Sigmoid()

        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_attention = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attention

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attention
        return x

# Double Conv Block
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# UNet Generator with CBAM
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(UNetGenerator, self).__init__()

        # Encoder
        self.down1 = double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.down5 = double_conv(512, 1024)

        # Bottleneck with CBAM
        self.cbam = CBAM(1024)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = double_conv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def _crop_and_concat(self, upsampled, skip):
        if upsampled.shape[2:] != skip.shape[2:]:
            diffY = skip.size(2) - upsampled.size(2)
            diffX = skip.size(3) - upsampled.size(3)
            skip = skip[:, :, diffY // 2 : skip.size(2) - (diffY - diffY // 2),
                              diffX // 2 : skip.size(3) - (diffX - diffX // 2)]
        return torch.cat((upsampled, skip), dim=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        d5 = self.down5(self.pool4(d4))

        bottleneck = self.cbam(d5)

        u1 = self.up1(bottleneck)
        u1 = self._crop_and_concat(u1, d4)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = self._crop_and_concat(u2, d3)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = self._crop_and_concat(u3, d2)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = self._crop_and_concat(u4, d1)
        u4 = self.conv4(u4)

        return self.final(u4)


# --- PatchGAN Discriminator ---
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            self._block(in_channels, 64, normalization=False),
            self._block(64, 128),
            self._block(128, 256),
            self._block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)  # Patch output
        )

    def _block(self, in_channels, out_channels, normalization=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
        if normalization:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)  # Concatenate grayscale input and color output
        return self.model(x)

# --- Custom Dataset Loader ---
class FluorescenceDataset(Dataset):
    def __init__(self, ch4_dir, ch1_dir, transform=None):
        self.ch4_dir = ch4_dir
        self.ch1_dir = ch1_dir
        self.transform = transform
        self.image_names = sorted([f for f in os.listdir(ch4_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        ch4_name = self.image_names[idx]
        ch1_name = ch4_name.replace("CH4", "CH1")
        
        ch4_img = Image.open(os.path.join(self.ch4_dir, ch4_name)).convert("L")  # Grayscale
        ch1_img = Image.open(os.path.join(self.ch1_dir, ch1_name)).convert("RGB")  # Color
        
        if self.transform:
            ch4_img = self.transform(ch4_img)
            ch1_img = self.transform(ch1_img)

        return ch4_img, ch1_img

# --- Losses ---
bce_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

def generator_loss(disc_pred_fake, fake_img, real_img, lambda_l1=100):
    adv_loss = bce_loss(disc_pred_fake, torch.ones_like(disc_pred_fake))
    l1 = l1_loss(fake_img, real_img)
    return adv_loss + lambda_l1 * l1

def discriminator_loss(disc_pred_real, disc_pred_fake):
    real_loss = bce_loss(disc_pred_real, torch.ones_like(disc_pred_real))
    fake_loss = bce_loss(disc_pred_fake, torch.zeros_like(disc_pred_fake))
    return (real_loss + fake_loss) * 0.5

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
#from Modelo import UNetGenerator, PatchDiscriminator, FluorescenceDataset, generator_loss, discriminator_loss
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch.nn.functional as F


def compute_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))

def compute_ssim(pred, target):
    pred_np = pred.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    target_np = target.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    score, _ = ssim(pred_np, target_np, full=True, channel_axis=2, data_range=1.0)
    return score


# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 300
batch_size = 4
lr = 1e-4
lambda_l1 = 100
save_dir = "checkpoints_Unet_Form"
output_dir = "train_outputs_unet_Form"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Dataset
transform = transforms.Compose([
    transforms.Resize((720, 960)),
    transforms.ToTensor()
])
#train_dataset = FluorescenceDataset(r"/mnt/DATA/Harold/data_with_fluorescence/chamber nonforming/CH4", "/mnt/DATA/Harold/data_with_fluorescence/chamber nonforming/CH1", transform)

from torch.utils.data import random_split, DataLoader

# Cargar el dataset completo
full_dataset = FluorescenceDataset(
    "/home/UNT/hel0057/Documents/Organoids/Chamber Forming/CH4",
    "/home/UNT/hel0057/Documents/Organoids/Chamber Forming/CH1",
    transform
)

# Calcular tamaños para train y val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Dividir el dataset
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Model
gen = UNetGenerator().to(device)
disc = PatchDiscriminator().to(device)

# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop
for epoch in range(epochs):
    if epoch == 0:
        metrics_list = []
    gen.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for idx, (gray, color) in enumerate(loop):
        gray, color = gray.to(device), color.to(device)

        # Train Discriminator
        fake_color = gen(gray)
        disc_real = disc(gray, color)
        disc_fake = disc(gray, fake_color.detach())
        d_loss = discriminator_loss(disc_real, disc_fake)

        opt_disc.zero_grad()
        d_loss.backward()
        opt_disc.step()

        # Train Generator
        disc_fake = disc(gray, fake_color)
        g_loss = generator_loss(disc_fake, fake_color, color, lambda_l1)

        opt_gen.zero_grad()
        g_loss.backward()
        opt_gen.step()

        loop.set_postfix(G_loss=g_loss.item(), D_loss=d_loss.item())

    # Save model and sample
    # Save model and sample
    torch.save(gen.state_dict(), f"{save_dir}/gen_epoch_{epoch+1}.pth")
    torch.save(disc.state_dict(), f"{save_dir}/disc_epoch_{epoch+1}.pth")
    
    
    gen.eval()
    with torch.no_grad():
        val_gray, val_color = next(iter(val_loader))
        val_gray, val_color = val_gray.to(device), val_color.to(device)
        pred_color = gen(val_gray)
        val_gray_3c = val_gray.repeat(1, 3, 1, 1)  # (B, 1, H, W) → (B, 3, H, W)
        # Selecciona solo las primeras 'n' imágenes para visualizar (por ejemplo, 4)
        # Selecciona solo las primeras 'n' imágenes para visualizar
        n = min(batch_size, 4)
        indices = random.sample(range(val_gray_3c.size(0)), n)
        val_gray_3c = val_gray_3c[indices]
        val_color = val_color[indices]
        pred_color = pred_color[indices]
        
        # Crea la comparación como (Input | Ground Truth | Predicción)
        comparison = torch.cat([
            val_gray_3c,
            val_color,
            pred_color
        ], dim=0)
        
        # Guarda la imagen combinada
        save_image(comparison, f"{output_dir}/comparison_epoch_{epoch+1}.png", nrow=n)
        # Calcular PSNR y SSIM promedio en el batch
        total_psnr = 0
        total_ssim = 0
        for i in range(n):
            psnr_val = compute_psnr(pred_color[i], val_color[i])
            ssim_val = compute_ssim(pred_color[i], val_color[i])
            total_psnr += psnr_val.item()
            total_ssim += ssim_val

        avg_psnr = total_psnr / n
        avg_ssim = total_ssim / n
        print(f"[Epoch {epoch+1}] PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
        metrics_list.append({
        "Epoch": epoch + 1,
        "Gen_Loss": g_loss.item(),
        "Dis_Loss": d_loss.item(),
        "PSNR": avg_psnr,
        "SSIM": avg_ssim
        })
        metrics_df = pd.DataFrame(metrics_list)
        os.makedirs(output_dir, exist_ok=True)
        metrics_df.to_excel(os.path.join(output_dir, "metrics_noform.xlsx"), index=False)
        
