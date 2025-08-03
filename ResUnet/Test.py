import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize as torch_resize
from torchvision import transforms
from Model.Model import ResUNetGenerator, Resnet50_classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from Utils.utils import get_brightfield_and_fluorescence_split
from torchvision.transforms.functional import to_pil_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ================= Transforms ============================
# Para ResNet50 (RGB)
transform_rgb_resnet = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Generador (ResUNet) 
transform_gray_resunet = T.Compose([
    transforms.Resize((720, 960)),
    transforms.ToTensor()
])

transform_raw = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])


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




# Output for Validation Images
output_dir = "/home/UNT/hel0057/Documents/Organoids/ResUnet/Validation"
os.makedirs(output_dir, exist_ok=True)

# Get data from the whole dataset and split 80% training / 20% validation
base_path = "/home/UNT/hel0057/Documents/Organoids"
X_train, X_val, Y_train, Y_val, y_train, y_val = get_brightfield_and_fluorescence_split(base_path, test_size=0.2)

# Evaluate and save
results = []


for i, (img_path, gt_path, label) in enumerate(tqdm(zip(X_val, Y_val, y_val), total=len(X_val))):
    # === 1. Classifier ===
    img_pil_rgb = Image.open(img_path).convert("RGB")
    img_tensor_rgb = transform_rgb_resnet(img_pil_rgb).unsqueeze(0).to(device)

    # === 2. Generator ===
    img_pil_gray = Image.open(img_path).convert("L")  # Tamaño original
    img_tensor_gray = transform_gray_resunet(img_pil_gray).unsqueeze(0).to(device)

    # === 3. Ground Truth ===
    gt_pil = Image.open(gt_path).convert("RGB")
    gt_tensor = T.ToTensor()(gt_pil).unsqueeze(0).to(device)  # No resize
    gt_np = gt_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # === 4. Classification ===
    pred = clf(img_tensor_rgb).item()
    pred_class = 1 if pred >= 0.5 else 0

    # === 5. Generation ===
    with torch.no_grad():
        if pred_class == 1:
            gen_tensor = G_forming(img_tensor_gray)
            model_name = "ResUNet_Forming"
        else:
            gen_tensor = G_nonforming(img_tensor_gray)
            model_name = "ResUNet_Nonforming"

    # Resize generated to ground truth size
    gt_height, gt_width = gt_pil.size[1], gt_pil.size[0]
    gen_tensor_resized = torch_resize(gen_tensor.squeeze(0), [gt_height, gt_width]).unsqueeze(0)
    gen_np = gen_tensor_resized.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # === 6. Metrics ===
    try:
        psnr_val = psnr(gt_np, gen_np, data_range=1.0)
        if gt_np.shape[-1] == 1:
            ssim_val = ssim(gt_np.squeeze(-1), gen_np.squeeze(-1), data_range=1.0)
        else:
            ssim_val = ssim(gt_np, gen_np, data_range=1.0, channel_axis=-1)
    except Exception as e:
        psnr_val, ssim_val = None, None
        print(f"⚠️ Error calculating image metrics {i}: {e}")

    # === 7. Save Results ===
    results.append({
        "Index": i,
        "Filename": os.path.basename(img_path),
        "True Label": label,
        "Predicted Label": pred_class,
        "Model Used": model_name,
        "PSNR": psnr_val,
        "SSIM": ssim_val,
    })

    # === 8. Save Visualization ===
    gen_pil = to_pil_image(gen_tensor_resized.squeeze(0).cpu())

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_pil_rgb)
    axs[0].set_title(f"Input (CH4)\nTrue: {'Forming' if label else 'Nonforming'}")
    axs[0].axis('off')

    axs[1].imshow(gt_pil)
    axs[1].set_title("Ground Truth (CH1)")
    axs[1].axis('off')

    axs[2].imshow(gen_pil)
    axs[2].set_title(f"Generated\nPred: {'Forming' if pred_class else 'Nonforming'}\n{model_name}")
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"visual_{i:03d}_{model_name}.png"))
    plt.close(fig)
    
# === Global Metrics ===
true_labels = [r["True Label"] for r in results]
predicted_labels = [r["Predicted Label"] for r in results]

acc = accuracy_score(true_labels, predicted_labels)
prec = precision_score(true_labels, predicted_labels)
rec = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print("\n📊 Clasificación ResNet50:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# (Opcional) Mostrar matriz de confusión
cm = confusion_matrix(true_labels, predicted_labels)
print("\nMatriz de confusión:")
print(cm)

summary_df = pd.DataFrame([{
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1 Score": f1,
    "Confusion matrix": cm
}])

#Save 2 files: Classification Report, Image Generation Metrics.

results_df1 = pd.DataFrame(summary_df)
results_df1.to_excel("/home/UNT/hel0057/Documents/Organoids/ResUnet/Validation/class_metrics.xlsx", index=False)
print("✅ Resultados guardados en class_metrics.xlsx")


# Guardar métricas en Excel
results_df = pd.DataFrame(results)
results_df.to_excel("/home/UNT/hel0057/Documents/Organoids/ResUnet/Validation/validation_metrics.xlsx", index=False)
print("✅ Resultados guardados en validation_metrics.xlsx")
