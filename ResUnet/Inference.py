import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize as torch_resize
from torchvision import transforms
from Model.Model import ResUNetGenerator, Resnet50_classifier
from torchvision.transforms.functional import to_pil_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# Load Resnet50
resnet_path = "Files/resnet50_CH4_best.pth"
#resnet_path = "/home/UNT/hel0057/Documents/Organoids/Classification/resnet50_CH4_best.pth"
clf = Resnet50_classifier(weights_path=resnet_path, device=device)


# Load ResUnet Gen_Chamber Forming
#G_chamber_path = "/home/UNT/hel0057/Documents/Organoids/ResUnet/Chamber Forming Results/Gen_Form/gen_epoch_111.pth"
G_chamber_path = "Files/gen_epoch_111.pth"
G_forming = ResUNetGenerator().to(device)
G_forming.load_state_dict(torch.load(G_chamber_path, map_location=device))
G_forming.eval()


# Load ResUnet Gen_Chamber NonForming
#G_nonchamber_path = "/home/UNT/hel0057/Documents/Organoids/ResUnet/Chamber NonForming Results/Gen_NoForm/gen_epoch_104.pth"
G_nonchamber_path = "Files/gen_epoch_104.pth"
G_nonforming = ResUNetGenerator().to(device)
G_nonforming.load_state_dict(torch.load(G_nonchamber_path, map_location=device))
G_nonforming.eval()



# === Image Path ===
#image_path = "/home/UNT/hel0057/Documents/Organoids/Chamber Forming/CH4/Image_XY01_CH4.tif" 
#image_path = "Image_XY02_CH4 (7).tif" 

from tkinter import Tk
from tkinter.filedialog import askopenfilename
Tk().withdraw()
image_path = askopenfilename(title="Select a CH4 image (brightfield)")

# Transform for Resnet50
transform_rgb_resnet = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Generator (ResUNet) 
transform_gray_resunet = T.Compose([
    transforms.Resize((720, 960)),
    transforms.ToTensor()
])

transform_raw = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# === Classifier (Resnet50):  RGB 256x256 ===
img_pil_rgb = Image.open(image_path).convert("RGB")
img_tensor_rgb = transform_rgb_resnet(img_pil_rgb).unsqueeze(0).to(device)

# === Generator: GRAYSCALE ===
img_pil_gray = Image.open(image_path).convert("L")
img_tensor_gray = transform_gray_resunet(img_pil_gray).unsqueeze(0).to(device)

# === Class Prediction ===
with torch.no_grad():
    pred = clf(img_tensor_rgb).item()
    pred_class = 1 if pred >= 0.5 else 0
    predicted_label = "Forming" if pred_class else "Nonforming"

# === Generation with the appropiate model ===
with torch.no_grad():
    if pred_class == 1:
        gen_tensor = G_forming(img_tensor_gray)
        model_name = "ResUNet_Forming"
    else:
        gen_tensor = G_nonforming(img_tensor_gray)
        model_name = "ResUNet_Nonforming"

# === Resize generated output to original size ===
orig_width, orig_height = img_pil_gray.size
gen_tensor_resized = torch_resize(gen_tensor.squeeze(0), [orig_height, orig_width]).unsqueeze(0)

# === Convert to PIL image for display ===
gen_pil = to_pil_image(gen_tensor_resized.squeeze(0).cpu())

# === Visualization ===
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].imshow(img_pil_rgb)
axs[0].set_title("Input (Brightfield - CH4)")
axs[0].axis('off')

axs[1].imshow(gen_pil)
axs[1].set_title(f"\nPredicted: {predicted_label}\nModel: {model_name}")
axs[1].axis('off')

plt.tight_layout()
plt.show()