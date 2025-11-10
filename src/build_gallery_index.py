# All comments in English.
import os, json, sys
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ResNet50Classifier

GALLERY_DIR = "data/polyvore"  # Use full Polyvore dataset
RESULTS_DIR = "results"
MODEL_PATH  = os.path.join(RESULTS_DIR, "model_best.pth")
CLASSMAP    = os.path.join(RESULTS_DIR, "class_to_idx.json")
BATCH_SIZE  = 32  # Process multiple images at once for speed

def collect_image_paths(root):
    exts = {'.jpg','.jpeg','.png'}
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn.lower())[1] in exts:
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)

class ImageDataset(Dataset):
    """Simple dataset for batch processing images"""
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor, img_path, True
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy tensor for failed images
            return torch.zeros(3, 224, 224), img_path, False

@torch.no_grad()
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Device selection: prioritize MPS (Apple Silicon), then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        print("Using CPU")

    # Load model
    with open(CLASSMAP, 'r') as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)

    model = ResNet50Classifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()

    # Transform
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Collect all image paths
    print("Collecting image paths...")
    img_paths = collect_image_paths(GALLERY_DIR)
    print(f"Found {len(img_paths)} images")

    # Create dataset and dataloader for batch processing
    dataset = ImageDataset(img_paths, tfm)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,  # Parallel data loading
        pin_memory=(device != "cpu")
    )

    # Extract features in batches
    feats_list, meta = [], []
    print("Extracting features...")
    
    for batch_imgs, batch_paths, batch_valid in tqdm(loader, desc="Processing batches"):
        batch_imgs = batch_imgs.to(device)
        _, feats = model(batch_imgs)  # (batch_size, 2048)
        feats_np = feats.cpu().numpy()
        
        for i, (img_path, is_valid) in enumerate(zip(batch_paths, batch_valid)):
            if is_valid:
                feats_list.append(feats_np[i])
                meta.append({"image_path": img_path})

    # Save results
    feats_mat = np.stack(feats_list, axis=0) if feats_list else np.zeros((0,2048), dtype=np.float32)
    np.savez(os.path.join(RESULTS_DIR, "gallery_index.npz"), feats=feats_mat)
    
    with open(os.path.join(RESULTS_DIR, "gallery_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nIndexing complete!")
    print(f"  Total images processed: {len(meta)}")
    print(f"  Feature matrix shape: {feats_mat.shape}")
    print(f"  Saved to: results/gallery_index.npz and gallery_meta.json")

if __name__ == "__main__":
    main()
