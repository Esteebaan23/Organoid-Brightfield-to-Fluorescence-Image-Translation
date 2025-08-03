import torch
import torch.nn as nn
from torchvision import transforms
import os
import random
import pandas as pd
from Model.Model import ResUNetGenerator, PatchDiscriminator
from Utils.utils import FluorescenceDataset
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader


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


# ====================== Config ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 300
batch_size = 4
lr = 1e-4
lambda_l1 = 100
save_dir = "checkpoints_ResUnet_NonForm"
output_dir = "train_outputs_Resunet_NonForm"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Dataset
transform = transforms.Compose([
    transforms.Resize((720, 960)),
    transforms.ToTensor()
])



# Cargar el dataset completo
full_dataset = FluorescenceDataset(
    "/home/UNT/hel0057/Documents/Organoids/Chamber Nonforming/CH4",
    "/home/UNT/hel0057/Documents/Organoids/Chamber Nonforming/CH1",
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
gen = ResUNetGenerator().to(device)
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
    torch.save(gen.state_dict(), f"{save_dir}/gen_epoch_{epoch+1}.pth")
    torch.save(disc.state_dict(), f"{save_dir}/disc_epoch_{epoch+1}.pth")
    
    
    gen.eval()
    with torch.no_grad():
        val_gray, val_color = next(iter(val_loader))
        val_gray, val_color = val_gray.to(device), val_color.to(device)
        pred_color = gen(val_gray)
        val_gray_3c = val_gray.repeat(1, 3, 1, 1)  # (B, 1, H, W) → (B, 3, H, W)
        n = min(batch_size, 4)
        indices = random.sample(range(val_gray_3c.size(0)), n)
        
        val_gray_3c = val_gray_3c[indices]
        val_color = val_color[indices]
        pred_color = pred_color[indices]
        
        # Create the comparison as (Input | Ground Truth | Prediction)
        comparison = torch.cat([
            val_gray_3c,
            val_color,
            pred_color
        ], dim=0)
        
        # Save Image
        save_image(comparison, f"{output_dir}/comparison_epoch_{epoch+1}.png", nrow=n)
        # Compute SSIM and PSNR
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
        metrics_df.to_excel(os.path.join(output_dir, "metrics.xlsx"), index=False)