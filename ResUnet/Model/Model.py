import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ========== Classifier (Resnet50) ==========



def Resnet50_classifier(weights_path=None, device='cpu'):
    clf = models.resnet50(weights=None)
    clf.fc = nn.Sequential(
        nn.Linear(clf.fc.in_features, 1),
        nn.Sigmoid()
    )

    if weights_path is not None:
        import torch
        state_dict = torch.load(weights_path, map_location=device)
        clf.load_state_dict(state_dict)

    return clf.to(device).eval()
    
    
# ========== Generator (ResUnet) ==========


# ========== CBAM Block ==========
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max_ = self.fc(self.max_pool(x))
        x = x * self.sigmoid(avg + max_)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.spatial(torch.cat([avg_out, max_out], dim=1))
        return x

# ========== Residual Block ==========
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


# ========== Res-UNet Generator ==========
class ResUNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.block(1, 64)
        self.enc2 = self.block(64, 128)
        self.enc3 = self.block(128, 256)
        self.cbam = CBAM(256)
        self.middle = ResidualBlock(256)
        self.dec3 = self.block(256 + 128, 128)
        self.dec2 = self.block(128 + 64, 64)
        self.final = nn.Conv2d(64, 3, 1)

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)  # -> 64x128x128
        e2 = self.enc2(e1)  # -> 128x64x64
        e3 = self.enc3(e2)  # -> 256x32x32
        e3 = self.cbam(e3)
        m = self.middle(e3)  # -> 256x32x32
    
        d3 = F.interpolate(m, size=e2.shape[2:])  # Asegura mismo tamaño que e2
        d3 = self.dec3(torch.cat([d3, e2], dim=1))  # -> 128x64x64
    
        d2 = F.interpolate(d3, size=e1.shape[2:])  # Asegura mismo tamaño que e1
        d2 = self.dec2(torch.cat([d2, e1], dim=1))  # -> 64x128x128
    
        out = F.interpolate(d2, size=x.shape[2:])  # Última upsampling a tamaño original
        out = torch.sigmoid(self.final(out))  # -> 3x256x256
        return out



# ========== PatchGAN Discriminator ==========
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(4, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))
    
