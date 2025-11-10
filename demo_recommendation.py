"""
AI-Powered Wardrobe Recommender - Demo Script
Demonstrates the recommendation system by taking a query image and finding similar items
"""
import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from src.model import ResNet50Classifier
from src.recommender import recommend_topk

# Configuration
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(RESULTS_DIR, "model_best.pth")
CLASSMAP = os.path.join(RESULTS_DIR, "class_to_idx.json")
GALLERY_INDEX = os.path.join(RESULTS_DIR, "gallery_index.npz")
GALLERY_META = os.path.join(RESULTS_DIR, "gallery_meta.json")


def load_model(device):
    """Load trained ResNet50 model"""
    with open(CLASSMAP, 'r') as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)

    model = ResNet50Classifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()

    return model, idx_to_class


def extract_query_features(image_path, model, device):
    """Extract features from query image"""
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, features = model(x)
        pred_class = logits.argmax(1).item()
        confidence = torch.softmax(logits, dim=1).max().item()

    return features.cpu().numpy(), pred_class, confidence


def visualize_recommendations(query_path, recommendations, idx_to_class, pred_class, confidence, save_path=None):
    """Visualize query image and top-k recommendations"""
    num_recs = len(recommendations)
    fig, axes = plt.subplots(1, num_recs + 1, figsize=(4 * (num_recs + 1), 4))

    # Show query image
    query_img = Image.open(query_path).convert('RGB')
    axes[0].imshow(query_img)
    axes[0].set_title(f"Query Image\nPredicted: {idx_to_class[str(pred_class)]}\nConfidence: {confidence:.2f}",
                      fontsize=10, fontweight='bold', color='blue')
    axes[0].axis('off')

    # Show recommendations
    for i, (meta, similarity) in enumerate(recommendations):
        rec_path = meta['image_path']
        rec_img = Image.open(rec_path).convert('RGB')
        axes[i + 1].imshow(rec_img)
        axes[i + 1].set_title(f"Rank {i+1}\nSimilarity: {similarity:.3f}",
                              fontsize=10, fontweight='bold', color='green')
        axes[i + 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()


def main():
    # Check if gallery index exists
    if not os.path.exists(GALLERY_INDEX):
        print("Gallery index not found!")
        print("Please run: python src/build_gallery_index.py")
        return

    # Device selection
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
    print("\nLoading model...")
    model, idx_to_class = load_model(device)
    print(f"Model loaded successfully! Classes: {list(idx_to_class.values())}")

    # Load gallery index
    print("\nLoading gallery index...")
    gallery_data = np.load(GALLERY_INDEX)
    gallery_feats = gallery_data['feats']

    with open(GALLERY_META, 'r') as f:
        gallery_meta = json.load(f)

    print(
        f"Gallery loaded: {gallery_feats.shape[0]} images with {gallery_feats.shape[1]}-dim features")

    # Example: Use first test image as query
    query_image = "data/deepfashion_subset/test/Blazer/img_00000001.jpg"

    # Allow user to specify custom query
    import sys
    if len(sys.argv) > 1:
        query_image = sys.argv[1]

    if not os.path.exists(query_image):
        print(f"Query image not found: {query_image}")
        print("Usage: python demo_recommendation.py [path_to_query_image]")
        return

    print(f"\nQuery image: {query_image}")

    # Extract query features
    print("Extracting query features...")
    query_feats, pred_class, confidence = extract_query_features(
        query_image, model, device)
    print(f"  Predicted class: {idx_to_class[str(pred_class)]}")
    print(f"  Confidence: {confidence:.4f}")

    # Get recommendations
    print("\nFinding similar items...")
    top_k = 5
    recommendations = recommend_topk(
        query_feats, gallery_feats, gallery_meta, k=top_k)

    print(f"\nTop {top_k} Recommendations:")
    for i, (meta, similarity) in enumerate(recommendations, 1):
        print(
            f"  {i}. {os.path.basename(meta['image_path'])} (similarity: {similarity:.4f})")

    # Visualize results
    print("\nGenerating visualization...")
    output_path = os.path.join(RESULTS_DIR, "recommendation_demo.png")
    visualize_recommendations(
        query_image,
        recommendations,
        idx_to_class,
        pred_class,
        confidence,
        save_path=output_path
    )

    print("\nRecommendation demo completed!")


if __name__ == "__main__":
    main()
