#!/usr/bin/env python3
"""
 CLIP Embeddings
 CLIP 
"""

import json
import torch
import clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np


def load_clip_model(device='cpu'):
    """ CLIP """
    print(" Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f" CLIP loaded on {device}")
    return model, preprocess


def encode_image(image_path, model, preprocess, device='cpu'):
    """"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            # 
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()[0]
    except Exception as e:
        print(f" Error encoding {image_path}: {e}")
        return None


def rebuild_embeddings():
    """ CLIP embeddings"""
    print("=" * 60)
    print(" CLIP Embeddings")
    print("=" * 60)

    #  MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        device = "mps"
        print(" Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = "cuda"
        print(" Using CUDA")
    else:
        device = "cpu"
        print(" Using CPU")

    #  CLIP
    model, preprocess = load_clip_model(device)

    # 
    print("\n Loading gallery metadata...")
    with open('results/gallery_meta.json', 'r', encoding='utf-8') as f:
        gallery = json.load(f)
    print(f" Loaded {len(gallery)} items")

    # 
    augmented_count = sum(
        1 for item in gallery if item.get('source') == 'augmented')
    original_count = len(gallery) - augmented_count
    print(f"  : {original_count} ")
    print(f"  : {augmented_count} ")

    #  embeddings ()
    embeddings_path = Path('results/gallery_embeddings.npy')
    if embeddings_path.exists():
        print(f"\n Loading existing embeddings...")
        existing_embeddings = np.load(embeddings_path)
        print(f" Loaded {len(existing_embeddings)} existing embeddings")

        # ,
        if len(existing_embeddings) == len(gallery):
            print("\n  Embeddings ")
            response = input(" embeddings? (y/N): ")
            if response.lower() != 'y':
                print("  embeddings")
                return
            rebuild_all = True
        else:
            rebuild_all = False
            print(f" {len(gallery) - len(existing_embeddings)} ")
    else:
        print("\n No existing embeddings found, building from scratch...")
        existing_embeddings = None
        rebuild_all = True

    # 
    print(f"\n Encoding images...")
    embeddings = []

    if rebuild_all:
        # 
        for item in tqdm(gallery, desc="Encoding all images"):
            image_path = item['image_path']
            embedding = encode_image(image_path, model, preprocess, device)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                # 
                embeddings.append(np.zeros(512))
    else:
        # 
        embeddings = list(existing_embeddings)
        start_idx = len(existing_embeddings)

        for item in tqdm(gallery[start_idx:], desc="Encoding new images"):
            image_path = item['image_path']
            embedding = encode_image(image_path, model, preprocess, device)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                embeddings.append(np.zeros(512))

    #  numpy array
    embeddings = np.array(embeddings)

    # 
    print(f"\n Saving embeddings...")

    #  embeddings
    if embeddings_path.exists():
        backup_path = Path('results/gallery_embeddings_backup.npy')
        import shutil
        shutil.copy(embeddings_path, backup_path)
        print(f"  embeddings  {backup_path}")

    # 
    np.save(embeddings_path, embeddings)
    print(f" Embeddings  {embeddings_path}")
    print(f"   Shape: {embeddings.shape}")

    print("\n" + "=" * 60)
    print(" Embeddings !")
    print("=" * 60)
    print("\n :")
    print("  1. : python3 evaluate_system.py")
    print("  2. : 65.31% → 67-68% (+2-3%)")
    print("=" * 60)


if __name__ == '__main__':
    rebuild_embeddings()
