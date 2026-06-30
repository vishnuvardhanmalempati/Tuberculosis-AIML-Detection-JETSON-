import os
import zipfile
import urllib.request
import shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DATASET_URL = "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
CACHE_PATH = "C:/Users/vishn/.cache/kagglehub/datasets/raddar/tuberculosis-chest-xrays-montgomery/versions/1/images/images"

def create_synthetic_dataset(output_dir):
    """
    Generates a synthetic chest X-ray dataset for fallback execution.
    """
    output_dir = Path(output_dir)
    cxr_dir = output_dir / "CXR_png"
    cxr_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic Chest X-ray dataset in {cxr_dir}...")
    for i in range(1, 101):
        label = 0 if i <= 50 else 1
        filename = f"MCUCXR_{i:04d}_{label}.png"
        save_path = cxr_dir / filename
        
        img = np.zeros((400, 400), dtype=np.uint8)
        cv2.ellipse(img, (200, 230), (150, 160), 0, 0, 360, 80, -1)
        cv2.ellipse(img, (135, 210), (45, 100), 0, 0, 360, 25, -1)
        cv2.ellipse(img, (265, 210), (45, 100), 0, 0, 360, 25, -1)
        cv2.ellipse(img, (200, 240), (35, 60), 0, 0, 360, 150, -1)
        cv2.line(img, (200, 70), (200, 370), 160, 6)
        
        for y in range(130, 310, 30):
            cv2.ellipse(img, (135, y), (55, 12), -12, 0, 180, 130, 2)
            cv2.ellipse(img, (265, y), (55, 12), 12, 180, 360, 130, 2)
            
        cv2.ellipse(img, (120, 105), (75, 8), 12, 0, 360, 190, 3)
        cv2.ellipse(img, (280, 105), (75, 8), -12, 0, 360, 190, 3)
        
        if label == 1:
            if i % 2 == 0:
                cv2.circle(img, (125, 160), 12, 160, -1)
                cv2.circle(img, (140, 175), 8, 140, -1)
            else:
                cv2.circle(img, (260, 260), 15, 150, -1)
                cv2.circle(img, (275, 245), 10, 130, -1)
                
        img = cv2.GaussianBlur(img, (21, 21), 0)
        noise = np.random.normal(0, 4, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(str(save_path), img)
        
    print(f"Synthetic dataset creation complete. Generated {i} images.")
    return output_dir

def download_and_extract_dataset(data_dir="data"):
    """
    Downloads/checks Montgomery County dataset. Prioritizes local kagglehub cache.
    """
    # 1. Check if the local kaggle cache path exists
    local_cache = Path(CACHE_PATH)
    if local_cache.exists():
        print(f"[INFO] Found pre-downloaded clinical dataset at: {local_cache}")
        return local_cache
        
    # 2. Check if extracted dataset already exists locally
    data_dir = Path(data_dir)
    extract_path = data_dir / "extracted"
    if (extract_path / "MontgomerySet").exists():
        print("Dataset already exists in extracted directory.")
        return extract_path / "MontgomerySet"
        
    # 3. Check if synthetic fallback already exists
    synthetic_path = data_dir / "synthetic"
    if (synthetic_path / "CXR_png").exists():
        print("Synthetic dataset already exists. Using synthetic fallback.")
        return synthetic_path
        
    # 4. Try downloading
    zip_path = data_dir / "NLM-MontgomeryCXRSet.zip"
    print(f"Attempting to download dataset from {DATASET_URL}...")
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rDownloading: {percent}% completed", end="")
        
        urllib.request.urlretrieve(DATASET_URL, zip_path, reporthook=progress_hook)
        print("\nDownload finished.")
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction completed.")
        
        if zip_path.exists():
            os.remove(zip_path)
            
        return extract_path / "MontgomerySet"
        
    except Exception as e:
        print(f"\n[WARNING] Download failed: {e}")
        print("Switching to synthetic dataset fallback.")
        if zip_path.exists():
            os.remove(zip_path)
        return create_synthetic_dataset(synthetic_path)

def get_image_paths_and_labels(dataset_path):
    """
    Scans the folder and extracts labels from the filename.
    Handles nested 'CXR_png' folder or direct file structure.
    """
    dataset_path = Path(dataset_path)
    cxr_dir = dataset_path / "CXR_png"
    
    # If the images are in the main directory directly (like the cached kaggle folder)
    if not cxr_dir.exists():
        cxr_dir = dataset_path
        
    image_files = list(cxr_dir.glob("*.png"))
    data = []
    
    for img_path in image_files:
        name = img_path.name
        if name.endswith("_0.png"):
            label = 0  # Normal
        elif name.endswith("_1.png"):
            label = 1  # Tuberculosis
        else:
            continue
        data.append((str(img_path), label))
        
    print(f"Total valid samples found: {len(data)}")
    normals = sum(1 for _, l in data if l == 0)
    tbs = sum(1 for _, l in data if l == 1)
    print(f"Normal: {normals}, Tuberculosis: {tbs}")
    return data

def split_dataset(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits the data into stratified Train, Val, and Test sets.
    """
    paths, labels = zip(*data)
    paths = list(paths)
    labels = list(labels)
    
    train_paths, val_test_paths, train_labels, val_test_labels = train_test_split(
        paths, labels, test_size=(val_ratio + test_ratio), random_state=seed, stratify=labels
    )
    
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        val_test_paths, val_test_labels, test_size=relative_test_ratio, random_state=seed, stratify=val_test_labels
    )
    
    train_data = list(zip(train_paths, train_labels))
    val_data = list(zip(val_paths, val_labels))
    test_data = list(zip(test_paths, test_labels))
    
    print(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

class TuberculosisDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[WARNING] Error reading file {img_path}: {e}")
            img = Image.new('RGB', (224, 224), color=0)
            
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.float32)

def get_dataloaders(data_dir="data", batch_size=16, seed=42):
    """
    Loads dataset and splits.
    """
    dataset_path = download_and_extract_dataset(data_dir)
    data = get_image_paths_and_labels(dataset_path)
    train_data, val_data, test_data = split_dataset(data, seed=seed)
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = TuberculosisDataset(train_data, transform=train_transform)
    val_dataset = TuberculosisDataset(val_data, transform=val_transform)
    test_dataset = TuberculosisDataset(test_data, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_data, val_data, test_data
