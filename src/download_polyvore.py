"""
Download Polyvore Outfit Dataset from Kaggle

Dataset: https://www.kaggle.com/datasets/enisteper1/polyvore-outfit-dataset

Dataset Features:
- 252,068 outfit images
- Outfit combinations and compatibility data
- Multiple categories (tops, bottoms, shoes, accessories)
- JSON metadata with outfit information

This script will:
1. Download Polyvore dataset from Kaggle using API
2. Extract and organize files
3. Keep dataset ready for outfit recommendation system
"""

import os
import sys
import shutil
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
POLYVORE_DIR = DATA_DIR / "polyvore"

# Kaggle dataset identifier
KAGGLE_DATASET = "enisteper1/polyvore-outfit-dataset"


def ensure_kaggle_setup():
    """Check if Kaggle API is properly configured"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        print("Error: Kaggle API credentials not found!")
        print("\nSetup Instructions:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Click 'Create New API Token'")
        print("  3. Move downloaded kaggle.json to ~/.kaggle/")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

    print("Kaggle API credentials found")
    return True


def download_polyvore():
    """Download Polyvore dataset from Kaggle"""

    print("=" * 80)
    print("  Downloading Polyvore Outfit Dataset from Kaggle")
    print("=" * 80)

    print("\nDataset Information:")
    print("  - 252,068 outfit images")
    print("  - Outfit compatibility data")
    print("  - Multiple categories (tops, bottoms, shoes, accessories)")
    print(f"  - Dataset: {KAGGLE_DATASET}")
    print("  - Size: ~2.8 GB")

    if not ensure_kaggle_setup():
        return False

    # Create directory
    POLYVORE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading to {POLYVORE_DIR}...")
    print("This may take 10-20 minutes depending on your connection...")

    try:
        # Download using Kaggle API
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(POLYVORE_DIR),
            unzip=True
        )

        print("\nDownload complete.")

        # Check what was downloaded
        print("\nDownloaded contents:")
        for item in POLYVORE_DIR.iterdir():
            if item.is_dir():
                count = len(list(item.rglob("*.*")))
                print(f"   - {item.name}/  ({count:,} files)")
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"   - {item.name}  ({size_mb:.1f} MB)")

        return True

    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        return False


def check_dataset():
    """Check if Polyvore dataset exists and show statistics"""

    if not POLYVORE_DIR.exists():
        print(f"Polyvore dataset not found at: {POLYVORE_DIR}")
        print("\nRun: python src/download_polyvore.py --download")
        return False

    print("=" * 80)
    print("  Polyvore Dataset Status")
    print("=" * 80)

    print(f"\nDataset location: {POLYVORE_DIR}")

    # Count files
    total_files = 0
    print("\nDataset contents:")
    for item in sorted(POLYVORE_DIR.iterdir()):
        if item.is_dir():
            count = len(list(item.rglob("*.*")))
            total_files += count
            print(f"   - {item.name}/  ({count:,} files)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"   - {item.name}  ({size_mb:.1f} MB)")

    print(f"\nTotal files: {total_files:,}")
    print("Polyvore dataset ready for use.")

    return True


def clean_temp_files():
    """Remove temporary zip files to save space"""

    print("\nCleaning temporary files...")

    removed_size = 0
    for zip_file in POLYVORE_DIR.glob("*.zip"):
        size_mb = zip_file.stat().st_size / (1024 * 1024)
        removed_size += size_mb
        zip_file.unlink()
        print(f"   Removed: {zip_file.name} ({size_mb:.1f} MB)")

    if removed_size > 0:
        print(f"\nFreed up {removed_size:.1f} MB of disk space")
    else:
        print("No temporary files found")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Polyvore Outfit Dataset from Kaggle"
    )
    parser.add_argument("--download", action="store_true",
                        help="Download Polyvore dataset from Kaggle")
    parser.add_argument("--check", action="store_true",
                        help="Check if dataset exists and show statistics")
    parser.add_argument("--clean", action="store_true",
                        help="Remove temporary zip files to save space")

    args = parser.parse_args()

    if args.download:
        if download_polyvore():
            print("\nNext steps:")
            print("   - Run: python src/download_polyvore.py --clean")
            print("   - Use dataset for outfit recommendation system")
    elif args.check:
        check_dataset()
    elif args.clean:
        clean_temp_files()
    else:
        # Show default help
        print("=" * 80)
        print("  Polyvore Outfit Dataset Downloader")
        print("=" * 80)

        print("\nDataset: Polyvore Outfit Dataset")
        print(f"   Source: Kaggle ({KAGGLE_DATASET})")
        print("   Size: ~2.8 GB")
        print("   Images: 252,068")

        print("\nUsage:")
        print("\n   1. Download dataset:")
        print("      python src/download_polyvore.py --download")

        print("\n   2. Check dataset status:")
        print("      python src/download_polyvore.py --check")

        print("\n   3. Clean temporary files:")
        print("      python src/download_polyvore.py --clean")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
