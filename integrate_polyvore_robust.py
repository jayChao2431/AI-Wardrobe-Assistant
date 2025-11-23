"""
 Polyvore  - 

"""

import json
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from pathlib import Path
import pickle

#  -  100 
BATCH_SIZE = 100
CHECKPOINT_FILE = 'results/polyvore_checkpoint.pkl'
OUTPUT_META = 'results/gallery_meta_polyvore.json'
OUTPUT_FEATURES = 'results/gallery_index_polyvore.npz'


def load_model():
    """ ResNet50"""
    device = torch.device('cpu')  #  CPU  MPS 
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model, device


def validate_image(img_path):
    """"""
    try:
        img = Image.open(img_path)
        img.verify()  # 
        # (verify )
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        # 
        if w < 50 or h < 50 or w > 5000 or h > 5000:
            return False
        return True
    except:
        return False


def extract_features_safe(model, image_path, device):
    """ -  timeout """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(img_tensor)

        return features.cpu().numpy().flatten()
    except:
        return None


def load_checkpoint():
    """"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return {'processed_ids': set(), 'gallery_meta': [], 'features_list': []}


def save_checkpoint(data):
    """"""
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(data, f)


def map_category(semantic_cat, description):
    """"""
    desc_lower = description.lower()

    if semantic_cat == 'tops':
        if 'blazer' in desc_lower or 'jacket' in desc_lower:
            return 'Blazer'
        elif 'blouse' in desc_lower:
            return 'Blouse'
        else:
            return 'Tee'
    elif semantic_cat == 'bottoms':
        if 'skirt' in desc_lower:
            return 'Skirt'
        else:
            return None  # 
    elif semantic_cat == 'all-body':
        if 'dress' in desc_lower:
            return 'Dress'
    return None


def infer_gender(description, categories):
    """"""
    text = (description + ' ' + ' '.join(categories)).lower()
    if any(x in text for x in ["women's", 'women', 'female', 'lady', 'girl']):
        return 'Female'
    elif any(x in text for x in ["men's", 'men', 'male', 'gentleman', 'boy']):
        return 'Male'
    return 'Unisex'


def main():
    print("=== Polyvore  -  ===\n")

    #  metadata
    print(" Polyvore metadata...")
    with open('datasets/item_metadata.json') as f:
        metadata = json.load(f)

    # 
    print("...")
    clothing_items = []
    for item in metadata:
        semantic_cat = item.get('semantic_category', '')
        if semantic_cat in ['tops', 'bottoms', 'all-body']:
            our_category = map_category(
                semantic_cat, item.get('description', ''))
            if our_category:
                item['our_category'] = our_category
                item['our_gender'] = infer_gender(
                    item.get('description', ''),
                    item.get('categories', [])
                )
                clothing_items.append(item)

    print(f"  {len(clothing_items)} ")

    #  3000  ()
    import random
    random.shuffle(clothing_items)
    clothing_items = clothing_items[:3000]

    print(f"  {len(clothing_items)} \n")

    # 
    checkpoint = load_checkpoint()
    print(f": {len(checkpoint['processed_ids'])} ")

    # 
    print(" ResNet50...")
    model, device = load_model()
    print(f" : {device}\n")

    # 
    print("...")
    skipped = 0
    batch_count = 0

    for item in tqdm(clothing_items):
        item_id = item['item_id']

        # 
        if item_id in checkpoint['processed_ids']:
            continue

        img_path = f"datasets/images/{item_id}.jpg"

        # 
        if not os.path.exists(img_path) or not validate_image(img_path):
            skipped += 1
            continue

        # 
        features = extract_features_safe(model, img_path, device)
        if features is None:
            skipped += 1
            continue

        #  gallery
        checkpoint['gallery_meta'].append({
            'image_path': img_path,
            'category': item['our_category'],
            'gender': item['our_gender'],
            'source': 'polyvore',
            'title': item.get('title', ''),
            'description': item.get('description', '')[:100]
        })
        checkpoint['features_list'].append(features)
        checkpoint['processed_ids'].add(item_id)

        #  BATCH_SIZE 
        batch_count += 1
        if batch_count % BATCH_SIZE == 0:
            save_checkpoint(checkpoint)

    print(f"\n : {len(checkpoint['gallery_meta'])} ")
    print(f" : {skipped} ")

    # 
    print("\n Gallery...")
    with open(OUTPUT_META, 'w') as f:
        json.dump(checkpoint['gallery_meta'], f, indent=2)

    features_array = np.array(checkpoint['features_list'])
    np.savez_compressed(OUTPUT_FEATURES, features=features_array)

    print(f" Gallery Meta: {OUTPUT_META}")
    print(f" Gallery Features: {OUTPUT_FEATURES}")
    print(f" : {features_array.shape}")

    # 
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print("\n=== ! ===")


if __name__ == '__main__':
    main()
