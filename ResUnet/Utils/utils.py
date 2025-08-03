import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image

# --- Custom Dataset Loader --- Covers the whole dataset
def get_brightfield_and_fluorescence_split(base_path, test_size=0.2):
    X, Y, labels = [], [], []

    for label_str, label_num in [("Chamber Forming", 1), ("Chamber Nonforming", 0)]:
        ch4_path = os.path.join(base_path, label_str, "CH4")
        ch1_path = os.path.join(base_path, label_str, "CH1")

        files = sorted([f for f in os.listdir(ch4_path) if f.endswith((".png", ".jpg", ".jpeg", ".tif"))])
        for f in files:
            ch4_img = os.path.join(ch4_path, f)
            ch1_img = os.path.join(ch1_path, f.replace("CH4", "CH1"))
            if os.path.exists(ch1_img):
                X.append(ch4_img)
                Y.append(ch1_img)
                labels.append(label_num)

    return train_test_split(X, Y, labels, test_size=test_size, stratify=labels, random_state=23)


# --- Custom Dataset Loader --- Only for one class.
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