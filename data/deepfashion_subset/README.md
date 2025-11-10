# DeepFashion Subset

This directory contains a curated subset of the DeepFashion Category and Attribute Prediction dataset for efficient training and experimentation.

## Dataset Statistics

### Full Dataset (organized)
- Total Images: 20,000
- Categories: 45 fashion categories
- Split Distribution:
  - Training: 14,000 images
  - Validation: 2,000 images
  - Test: 4,000 images

### Subset (for quick training)
- Total Images: 821
- Categories: 5 (Blazer, Blouse, Dress, Skirt, Tee)
- Split Distribution:
  - Training: 521 images (70%)
  - Validation: 150 images (15%)
  - Test: 150 images (15%)

## Category Breakdown

| Category | Train | Val | Test | Total |
|----------|-------|-----|------|-------|
| Blazer   | 131   | 30  | 30   | 191   |
| Blouse   | 140   | 30  | 30   | 200   |
| Dress    | 93    | 30  | 30   | 153   |
| Skirt    | 76    | 30  | 30   | 136   |
| Tee      | 81    | 30  | 30   | 141   |

## Directory Structure

```
deepfashion_subset/
├── train/
│   ├── Blazer/      (131 images)
│   ├── Blouse/      (140 images)
│   ├── Dress/       (93 images)
│   ├── Skirt/       (76 images)
│   └── Tee/         (81 images)
├── val/
│   ├── Blazer/      (30 images)
│   ├── Blouse/      (30 images)
│   ├── Dress/       (30 images)
│   ├── Skirt/       (30 images)
│   └── Tee/         (30 images)
└── test/
    ├── Blazer/      (30 images)
    ├── Blouse/      (30 images)
    ├── Dress/       (30 images)
    ├── Skirt/       (30 images)
    └── Tee/         (30 images)
```

## Usage

### Load with PyTorch DataLoader

```python
from src.dataset_loader import get_dataloaders

train_loader, val_loader, test_loader, idx_to_class = get_dataloaders(
    'data/deepfashion_subset', 
    batch_size=16, 
    num_workers=2, 
    img_size=224
)

print(f"Classes: {list(idx_to_class.values())}")
print(f"Training samples: {len(train_loader.dataset)}")
```

### Training with Jupyter Notebook

Open and run `notebooks/train_and_evaluate_detailed.ipynb` which is already configured to use this subset.

## Recreate or Modify Subset

To create a different subset with custom parameters:

```bash
# Create subset with 10 categories and 300 samples each
python src/organize_deepfashion.py --subset --categories 10 --samples 300

# Reorganize full dataset and create new subset
python src/organize_deepfashion.py --all --categories 5 --samples 200
```

## Notes

- Images are in JPG format at various resolutions
- Data augmentation is applied during training (see `src/dataset_loader.py`)
- Categories selected based on highest image count in full dataset
- Random seed (42) ensures reproducible splits
