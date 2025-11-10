# Data Directory# Data Folder



This directory contains datasets for the AI Wardrobe Assistant project.This folder contains **sample images** for the AI-Wardrobe-Assistant project.



## Datasets## DeepFashion

- Source: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

### 1. DeepFashion (Category and Attribute Prediction)- Used for: clothing classification and attribute extraction

- **Purpose**: Train CNN classifier for clothing recognition- Samples included here are placeholders for Deliverable 1 verification only.

- **Size**: ~13 GB (full), ~100 MB (subset)
- **Categories**: 50 clothing types (Tee, Jeans, Dress, etc.)
- **Images**: 289,222
- **Download**: Manual from Google Drive (see instructions below)

### 2. Polyvore (Outfit Dataset)
- **Purpose**: Build outfit recommendation system
- **Size**: ~2.8 GB
- **Images**: 252,068 outfit images
- **Download**: Automatic from Kaggle

## Directory Structure

After setup, you should have:

```
data/
├── deepfashion/                    # Full DeepFashion dataset
│   ├── train/
│   │   ├── Tee/
│   │   ├── Jeans/
│   │   ├── Dress/
│   │   └── ... (50 categories)
│   ├── val/
│   └── test/
│
├── deepfashion_subset/             # Smaller subset for quick training
│   ├── train/
│   │   ├── Tee/
│   │   ├── Jeans/
│   │   ├── Dress/
│   │   ├── Sweater/
│   │   └── Hoodie/
│   ├── val/
│   └── test/
│
└── polyvore/                       # Polyvore outfit dataset
    └── (outfit images and metadata)
```

## Quick Start

### Download DeepFashion

```bash
# 1. Show download instructions
python src/download_deepfashion.py

# 2. After manual download, extract and organize
python src/download_deepfashion.py --extract
python src/download_deepfashion.py --organize
python src/download_deepfashion.py --subset

# Or run all steps at once
python src/download_deepfashion.py --all
```

### Download Polyvore

```bash
# Download from Kaggle (automatic)
python src/download_polyvore.py --download

# Clean temporary files
python src/download_polyvore.py --clean

# Check status
python src/download_polyvore.py --check
```

## Detailed Setup

For step-by-step instructions, see:
- **DeepFashion**: `DEEPFASHION_SETUP.md`
- **Polyvore**: Run `python src/download_polyvore.py` for help

## Notes

- ⚠️ DeepFashion requires manual download from Google Drive (~13 GB)
- ⚠️ Polyvore downloads automatically from Kaggle (requires API key)
- ⚠️ Keep both datasets for complete functionality
- 💡 Use `deepfashion_subset/` for quick training experiments
- 💡 Full `deepfashion/` dataset for production model training

## Disk Space Requirements

- DeepFashion: ~15 GB (including extracted files)
- Polyvore: ~3 GB
- Total: ~18 GB free space recommended
