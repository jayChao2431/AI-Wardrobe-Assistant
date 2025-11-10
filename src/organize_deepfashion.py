"""
Organize DeepFashion dataset and create subset for training

This script reads the annotation files from DeepFashion dataset
and organizes images into train/val/test folders by category.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import random

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEEPFASHION_DIR = DATA_DIR / "deepfashion"
SUBSET_DIR = DATA_DIR / "deepfashion_subset"

# Category mapping from list_category_cloth.txt
CATEGORY_NAMES = {
    1: "Anorak", 2: "Blazer", 3: "Blouse", 4: "Bomber", 5: "Button-Down",
    6: "Cardigan", 7: "Flannel", 8: "Halter", 9: "Henley", 10: "Hoodie",
    11: "Jacket", 12: "Jersey", 13: "Parka", 14: "Peacoat", 15: "Poncho",
    16: "Sweater", 17: "Tank", 18: "Tee", 19: "Top", 20: "Turtleneck",
    21: "Capris", 22: "Chinos", 23: "Culottes", 24: "Cutoffs", 25: "Gauchos",
    26: "Jeans", 27: "Jeggings", 28: "Jodhpurs", 29: "Joggers", 30: "Leggings",
    31: "Sarong", 32: "Shorts", 33: "Skirt", 34: "Sweatpants", 35: "Sweatshorts",
    36: "Trunks", 37: "Caftan", 38: "Cape", 39: "Coat", 40: "Coverup",
    41: "Dress", 42: "Jumpsuit", 43: "Kaftan", 44: "Kimono", 45: "Nightdress",
    46: "Onesie", 47: "Robe", 48: "Romper", 49: "Shirtdress", 50: "Sundress"
}


def organize_by_category():
    """Organize images into train/val/test folders by category"""
    
    print("\nOrganizing DeepFashion dataset by category...")
    print("=" * 70)
    
    anno_dir = DEEPFASHION_DIR / "Anno_fine"
    img_dir = DEEPFASHION_DIR / "img"
    
    if not anno_dir.exists():
        print(f"Error: Annotation directory not found: {anno_dir}")
        return False
    
    # Process each split
    splits = {
        'train': ('train.txt', 'train_cate.txt'),
        'val': ('val.txt', 'val_cate.txt'),
        'test': ('test.txt', 'test_cate.txt')
    }
    
    stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    
    for split_name, (img_file, cate_file) in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Read image paths
        img_list_path = anno_dir / img_file
        with open(img_list_path, 'r') as f:
            img_paths = [line.strip() for line in f.readlines()]
        
        # Read categories
        cate_list_path = anno_dir / cate_file
        with open(cate_list_path, 'r') as f:
            categories = [int(line.strip()) for line in f.readlines()]
        
        if len(img_paths) != len(categories):
            print(f"Warning: Mismatch in {split_name} - images: {len(img_paths)}, categories: {len(categories)}")
            continue
        
        # Copy files to organized structure
        processed = 0
        for img_path, category_id in zip(img_paths, categories):
            src_file = DEEPFASHION_DIR / img_path
            
            if not src_file.exists():
                continue
            
            category_name = CATEGORY_NAMES.get(category_id, f"Category_{category_id}")
            dest_dir = DEEPFASHION_DIR / split_name / category_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            dest_file = dest_dir / src_file.name
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)
            
            stats[category_name][split_name] += 1
            processed += 1
            
            if processed % 1000 == 0:
                print(f"  Processed {processed:,} images...")
        
        print(f"  Completed: {processed:,} images")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Organization Complete")
    print("=" * 70)
    print(f"\n{'Category':<25} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 70)
    
    total_train = total_val = total_test = 0
    
    for category in sorted(stats.keys()):
        train_count = stats[category]["train"]
        val_count = stats[category]["val"]
        test_count = stats[category]["test"]
        total = train_count + val_count + test_count
        
        total_train += train_count
        total_val += val_count
        total_test += test_count
        
        print(f"{category:<25} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10}")
    
    print("-" * 70)
    print(f"{'TOTAL':<25} {total_train:<10} {total_val:<10} {total_test:<10} {total_train+total_val+total_test:<10}")
    
    print(f"\nImages organized at:")
    print(f"  {DEEPFASHION_DIR}/train/")
    print(f"  {DEEPFASHION_DIR}/val/")
    print(f"  {DEEPFASHION_DIR}/test/")
    
    return True


def create_subset(num_categories=5, samples_per_category=200):
    """Create a smaller subset for quick training"""
    
    print(f"\nCreating subset with {num_categories} categories...")
    print(f"Samples per category: {samples_per_category}")
    print("=" * 70)
    
    # Check if organized data exists
    train_dir = DEEPFASHION_DIR / "train"
    if not train_dir.exists():
        print("Error: Organized dataset not found!")
        print("Please run --organize first")
        return False
    
    # Count images per category
    category_counts = defaultdict(int)
    
    for split in ["train", "val", "test"]:
        split_dir = DEEPFASHION_DIR / split
        if split_dir.exists():
            for category_dir in split_dir.iterdir():
                if category_dir.is_dir():
                    count = len(list(category_dir.glob("*.jpg")))
                    category_counts[category_dir.name] += count
    
    # Select top N categories
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:num_categories]
    selected_categories = [cat for cat, count in top_categories]
    
    print(f"\nSelected categories:")
    for cat, count in top_categories:
        print(f"  - {cat}: {count:,} images")
    
    # Create subset
    print(f"\nBuilding subset...")
    
    subset_stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    
    for split in ["train", "val", "test"]:
        # Calculate samples for this split
        if split == "train":
            n_samples = int(samples_per_category * 0.7)
        elif split == "val":
            n_samples = int(samples_per_category * 0.15)
        else:  # test
            n_samples = samples_per_category - int(samples_per_category * 0.7) - int(samples_per_category * 0.15)
        
        for category in selected_categories:
            src_dir = DEEPFASHION_DIR / split / category
            dest_dir = SUBSET_DIR / split / category
            
            if not src_dir.exists():
                continue
            
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all images
            images = list(src_dir.glob("*.jpg"))
            
            # Random sample
            random.seed(42)
            if len(images) > n_samples:
                selected_images = random.sample(images, n_samples)
            else:
                selected_images = images
            
            # Copy files
            for img_path in selected_images:
                dest_path = dest_dir / img_path.name
                if not dest_path.exists():
                    shutil.copy2(img_path, dest_path)
                subset_stats[category][split] += 1
    
    # Print subset statistics
    print("\n" + "=" * 70)
    print("Subset Statistics")
    print("=" * 70)
    print(f"\n{'Category':<25} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 70)
    
    total_train = total_val = total_test = 0
    
    for category in sorted(subset_stats.keys()):
        train_count = subset_stats[category]["train"]
        val_count = subset_stats[category]["val"]
        test_count = subset_stats[category]["test"]
        total = train_count + val_count + test_count
        
        total_train += train_count
        total_val += val_count
        total_test += test_count
        
        print(f"{category:<25} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10}")
    
    print("-" * 70)
    print(f"{'TOTAL':<25} {total_train:<10} {total_val:<10} {total_test:<10} {total_train+total_val+total_test:<10}")
    
    print(f"\nSubset created at: {SUBSET_DIR}")
    print("Ready for training.")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Organize DeepFashion dataset and create training subset"
    )
    parser.add_argument("--organize", action="store_true",
                       help="Organize images by category into train/val/test")
    parser.add_argument("--subset", action="store_true",
                       help="Create a smaller subset for quick training")
    parser.add_argument("--categories", type=int, default=5,
                       help="Number of categories in subset (default: 5)")
    parser.add_argument("--samples", type=int, default=200,
                       help="Samples per category in subset (default: 200)")
    parser.add_argument("--all", action="store_true",
                       help="Run organize and subset in sequence")
    
    args = parser.parse_args()
    
    if args.all:
        print("Running complete pipeline...")
        if organize_by_category():
            create_subset(num_categories=args.categories, samples_per_category=args.samples)
    elif args.organize:
        organize_by_category()
    elif args.subset:
        create_subset(num_categories=args.categories, samples_per_category=args.samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
