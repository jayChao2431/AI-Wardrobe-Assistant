"""
 - 

:
1.  100 
2. 
3.  Precision, Recall, F1-score
4. 
"""

import json
import numpy as np
import random
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import clip
import torch


def load_gallery_data():
    """ gallery """
    print(" Loading gallery data...")

    with open('results/gallery_meta.json') as f:
        metadata = json.load(f)

    with open('results/class_to_idx.json') as f:
        idx_to_class = json.load(f)
        class_to_idx = {v: int(k) for k, v in idx_to_class.items()}

    print(f" Loaded {len(metadata)} items")
    return metadata, idx_to_class, class_to_idx


def create_test_set(metadata, n_samples=100, seed=42):
    """ ()"""
    print(f"\n Creating test set ({n_samples} samples)...")

    random.seed(seed)

    # 
    by_category = defaultdict(list)
    for idx, item in enumerate(metadata):
        by_category[item['category']].append(idx)

    # 
    test_indices = []
    samples_per_class = n_samples // len(by_category)

    for category, indices in by_category.items():
        if len(indices) >= samples_per_class:
            sampled = random.sample(indices, samples_per_class)
        else:
            sampled = indices
        test_indices.extend(sampled)
        print(f"  {category:10s}: {len(sampled)} samples")

    # 
    random.shuffle(test_indices)
    test_indices = test_indices[:n_samples]

    print(f" Test set: {len(test_indices)} samples")
    return test_indices


def classify_image(image_path, ensemble_classifier, smart_validator, metadata_dict):
    """ Ensemble Classifier + Smart Validator """
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f" Error loading {image_path}: {e}")
        return None, 0.0

    #  metadata ()
    text_info = ""
    if image_path in metadata_dict:
        meta = metadata_dict[image_path]
        title = meta.get('title', '')
        desc = meta.get('description', '')
        text_info = f"{title} {desc}".strip()

    # 
    pred_class, confidence, details = ensemble_classifier.classify(
        image=image,
        text_info=text_info,
        image_path=image_path,
        return_details=True
    )

    #  Smart Validator 
    all_scores = details.get('final_scores', {})
    final_class, final_confidence, validation_info = smart_validator.validate_classification(
        predicted_category=pred_class,
        confidence=confidence,
        all_scores=all_scores,
        text_info=text_info,
        image_path=image_path
    )

    return final_class, final_confidence


def evaluate_classification(metadata, test_indices, ensemble_classifier, smart_validator, class_to_idx):
    """"""
    print("\n🧪 Evaluating classification accuracy (Ensemble + Smart Validator)...")

    #  metadata  (image_path -> metadata)
    metadata_dict = {item['image_path']: item for item in metadata}

    y_true = []
    y_pred = []
    confidences = []
    errors = []

    for i, idx in enumerate(test_indices):
        item = metadata[idx]
        true_class = item['category']
        image_path = item['image_path']  # 

        pred_class, confidence = classify_image(
            image_path, ensemble_classifier, smart_validator, metadata_dict)

        if pred_class is None:
            continue

        y_true.append(class_to_idx[true_class])
        y_pred.append(class_to_idx[pred_class.capitalize()])  # 
        confidences.append(confidence)

        if pred_class != true_class:
            errors.append({
                'image': image_path,
                'true': true_class,
                'pred': pred_class,
                'confidence': confidence
            })

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(test_indices)}")

    print(f" Evaluated {len(y_true)} images")

    return np.array(y_true), np.array(y_pred), confidences, errors


def calculate_metrics(y_true, y_pred, idx_to_class):
    """"""
    print("\n Calculating metrics...")

    n_classes = len(idx_to_class)
    class_names = [idx_to_class[str(i)] for i in range(n_classes)]

    # 
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    # 
    metrics = {}
    for i, class_name in enumerate(class_names):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        tn = confusion.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': confusion[i, :].sum()
        }

    # 
    accuracy = np.trace(confusion) / confusion.sum()

    return accuracy, confusion, metrics


def plot_confusion_matrix(confusion, class_names, save_path='results/confusion_matrix.png'):
    """"""
    plt.figure(figsize=(10, 8))

    # 
    confusion_norm = confusion.astype(
        'float') / confusion.sum(axis=1)[:, np.newaxis]

    sns.heatmap(confusion_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Confusion matrix saved to {save_path}")
    plt.close()


def plot_category_distribution(metadata, save_path='results/category_distribution.png'):
    """"""
    categories = [item['category'] for item in metadata]
    counter = Counter(categories)

    plt.figure(figsize=(10, 6))
    plt.bar(counter.keys(), counter.values(),
            color='steelblue', edgecolor='black')
    plt.title('Category Distribution in Gallery (6,000 items)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)

    # 
    for cat, count in counter.items():
        plt.text(cat, count + 50, str(count), ha='center',
                 fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Category distribution saved to {save_path}")
    plt.close()


def generate_report(accuracy, metrics, errors, confidences, save_path='results/evaluation_report.txt'):
    """"""
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("AI Wardrobe Assistant - Classification Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")

        f.write("Per-Class Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"{'Category':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 60 + "\n")

        for cat, m in metrics.items():
            f.write(f"{cat:<12} {m['precision']:<12.3f} {m['recall']:<12.3f} "
                    f"{m['f1']:<12.3f} {m['support']:<10}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Average Confidence: {np.mean(confidences):.3f}\n")
        f.write(f"Total Errors: {len(errors)}\n\n")

        if errors:
            f.write("Sample Errors (first 10):\n")
            f.write("-" * 60 + "\n")
            for i, err in enumerate(errors[:10]):
                f.write(f"{i+1}. {err['image']}\n")
                f.write(f"   True: {err['true']}, Predicted: {err['pred']} "
                        f"(confidence: {err['confidence']:.3f})\n\n")

    print(f" Report saved to {save_path}")


def main():
    print("=" * 60)
    print("AI Wardrobe Assistant - System Evaluation")
    print(" A :  (CLIP 95% + Keyword 3% + Path 2%)")
    print("=" * 60)

    # 
    metadata, idx_to_class, class_to_idx = load_gallery_data()

    # 
    test_indices = create_test_set(metadata, n_samples=100)

    #  CLIP 
    print("\n Loading CLIP model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    print(f" CLIP loaded on {device}")

    #  + Smart Validator
    print("\n Initializing Ensemble Classifier + Smart Validator...")
    from src.ensemble_classifier import EnsembleClassifier
    from src.smart_validator import SmartValidator
    categories = ['blazer', 'blouse', 'dress',
                  'skirt', 'tee', 'pants', 'shorts']
    ensemble_classifier = EnsembleClassifier(
        clip_model=clip_model,
        clip_preprocess=preprocess,
        categories=categories,
        device=device
    )
    smart_validator = SmartValidator(categories=categories)
    print(" Ensemble Classifier ready (CLIP 95% + Keyword 3% + Path 2%)")
    print(" Smart Validator ready (thresholds: 0.90/0.70/0.50)")

    # 
    y_true, y_pred, confidences, errors = evaluate_classification(
        metadata, test_indices, ensemble_classifier, smart_validator, class_to_idx
    )

    # 
    accuracy, confusion, metrics = calculate_metrics(
        y_true, y_pred, idx_to_class)

    # 
    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n Overall Accuracy: {accuracy*100:.2f}%")
    print(f" Average Confidence: {np.mean(confidences):.3f}")
    print(f" Total Errors: {len(errors)}")

    print("\nPer-Class Performance:")
    print("-" * 60)
    for cat, m in metrics.items():
        print(f"{cat:10s}: Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, "
              f"F1={m['f1']:.3f}, Support={m['support']}")

    # 
    print("\n Generating visualizations...")
    class_names = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
    plot_confusion_matrix(confusion, class_names)
    plot_category_distribution(metadata)

    # 
    generate_report(accuracy, metrics, errors, confidences)

    print("\n" + "=" * 60)
    print(" Evaluation completed!")
    print("=" * 60)
    print("\n:")
    print("  - results/confusion_matrix.png")
    print("  - results/category_distribution.png")
    print("  - results/evaluation_report.txt")
    print("\n:")
    print("  1. ")
    print("  2.  FINAL_PROJECT_REPORT.md")
    print("  3.  ( 20 )")


if __name__ == '__main__':
    main()
