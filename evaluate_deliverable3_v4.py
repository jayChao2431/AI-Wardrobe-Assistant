"""
AI Wardrobe Assistant - Deliverable 3 Evaluation Script (v4.0)
Ensemble Classifier Performance Analysis
"""

from src.smart_validator import SmartValidator
from src.ensemble_classifier import EnsembleClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import clip
import torch
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# Import custom modules
sys.path.append(str(Path(__file__).parent.parent))

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Device configuration
device = "mps" if torch.backends.mps.is_available(
) else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}\n")

# Paths
DATA_ROOT = "data/deepfashion_subset"
RESULTS_DIR = "results"
CLASSMAP = os.path.join(RESULTS_DIR, "class_to_idx.json")

# Load CLIP model
print("Loading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Load class names
with open(CLASSMAP, 'r') as f:
    idx_to_class = json.load(f)
    class_to_idx = {v: int(k) for k, v in idx_to_class.items()}

num_classes = len(idx_to_class)
class_names = [idx_to_class[str(i)] for i in range(num_classes)]

print(f"Classes ({num_classes}): {class_names}\n")

# Initialize ensemble classifier and validator
ensemble = EnsembleClassifier(
    clip_model, clip_preprocess, class_names, device=device)
validator = SmartValidator(categories=class_names)

print("Ensemble Classifier loaded successfully!")
print(f"  - CLIP Model: ViT-B/32")
print(f"  - Components: CLIP 95% + Keyword 3% + Path 2%")
print(f"  - Smart Validator: Confidence-based (0.90/0.70/0.50)\n")

# ========== Load Test Data ==========
print("Loading test images...")
test_dir = os.path.join(DATA_ROOT, "test")
test_images = []
test_labels = []
test_paths = []

for class_name in class_names:
    class_dir = os.path.join(test_dir, class_name)
    if not os.path.exists(class_dir):
        continue

    for img_file in os.listdir(class_dir):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_dir, img_file)
            test_paths.append(img_path)
            test_labels.append(class_to_idx[class_name])

print(f"Found {len(test_paths)} test images")
print(f"\nClass distribution:")
for i, class_name in enumerate(class_names):
    count = test_labels.count(i)
    print(f"  {class_name}: {count}")

# ========== Generate Predictions ==========
print(f"\nGenerating predictions with Ensemble Classifier...")

y_true = []
y_pred = []
y_probs = []
predictions_detailed = []

for i, (img_path, true_label) in enumerate(zip(test_paths, test_labels)):
    if (i + 1) % 20 == 0:
        print(f"  Processing {i+1}/{len(test_paths)}...")

    try:
        # Load image
        image = Image.open(img_path).convert('RGB')

        # Get ensemble prediction
        result = ensemble.classify(
            image=image,
            image_path=img_path
        )

        # Validate
        validated = validator.validate_classification(
            result['predicted_class'],
            result['confidence'],
            result
        )

        pred_label = class_to_idx[validated['final_class']]

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_probs.append(validated['final_confidence'])

        predictions_detailed.append({
            'image_path': img_path,
            'true_class': class_names[true_label],
            'pred_class': validated['final_class'],
            'confidence': validated['final_confidence'],
            'clip_score': result['clip_score'],
            'keyword_score': result['keyword_score'],
            'path_score': result['path_score'],
            'correct': (pred_label == true_label)
        })

    except Exception as e:
        print(f"  Error processing {img_path}: {e}")
        continue

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)
predictions_df = pd.DataFrame(predictions_detailed)

print(f"\nPredictions complete: {len(y_true)} samples processed")

# ========== Calculate Metrics ==========
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None, zero_division=0
)

# Create metrics dataframe
metrics_df = pd.DataFrame({
    'Class': class_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

print(f"\n{'='*70}")
print(f"DELIVERABLE 3 - ENSEMBLE CLASSIFIER PERFORMANCE")
print(f"{'='*70}\n")
print(f"Model: CLIP ViT-B/32 + Keyword + Path Analysis")
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
print(metrics_df.to_string(index=False))
print(f"\n{'='*70}")

# Macro averages
print(f"\nMacro Averages:")
print(f"  Precision: {precision.mean():.4f}")
print(f"  Recall:    {recall.mean():.4f}")
print(f"  F1-Score:  {f1.mean():.4f}")

# Weighted averages
weighted_p = np.average(precision, weights=support)
weighted_r = np.average(recall, weights=support)
weighted_f1 = np.average(f1, weights=support)
print(f"\nWeighted Averages:")
print(f"  Precision: {weighted_p:.4f}")
print(f"  Recall:    {weighted_r:.4f}")
print(f"  F1-Score:  {weighted_f1:.4f}")

# ========== Confusion Matrix ==========
print(f"\nGenerating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0], cbar_kws={'label': 'Count'}, annot_kws={'size': 10})
axes[0].set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
axes[0].set_ylabel('True Label', fontweight='bold', fontsize=12)
axes[0].set_title('Confusion Matrix (Raw Counts)',
                  fontweight='bold', fontsize=14)
axes[0].tick_params(axis='x', rotation=45)

# Normalized
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1], cbar_kws={'label': 'Proportion'}, annot_kws={'size': 10})
axes[1].set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
axes[1].set_ylabel('True Label', fontweight='bold', fontsize=12)
axes[1].set_title('Confusion Matrix (Normalized)',
                  fontweight='bold', fontsize=14)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'deliverable3_v4_confusion_matrix.png'),
            dpi=300, bbox_inches='tight')
print(f"  Saved: deliverable3_v4_confusion_matrix.png")
plt.close()

# ========== Per-Class Metrics ==========
print(f"Generating per-class metrics...")
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# Precision
axes[0, 0].bar(class_names, precision, color='steelblue', alpha=0.8)
axes[0, 0].set_ylabel('Precision', fontweight='bold', fontsize=12)
axes[0, 0].set_title('Precision by Class', fontweight='bold', fontsize=14)
axes[0, 0].set_ylim([0, 1.05])
axes[0, 0].axhline(y=precision.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {precision.mean():.3f}')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Recall
axes[0, 1].bar(class_names, recall, color='coral', alpha=0.8)
axes[0, 1].set_ylabel('Recall', fontweight='bold', fontsize=12)
axes[0, 1].set_title('Recall by Class', fontweight='bold', fontsize=14)
axes[0, 1].set_ylim([0, 1.05])
axes[0, 1].axhline(y=recall.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {recall.mean():.3f}')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(axis='y', alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# F1-Score
axes[1, 0].bar(class_names, f1, color='seagreen', alpha=0.8)
axes[1, 0].set_ylabel('F1-Score', fontweight='bold', fontsize=12)
axes[1, 0].set_title('F1-Score by Class', fontweight='bold', fontsize=14)
axes[1, 0].set_ylim([0, 1.05])
axes[1, 0].axhline(y=f1.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {f1.mean():.3f}')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

# Support
axes[1, 1].bar(class_names, support, color='mediumpurple', alpha=0.8)
axes[1, 1].set_ylabel('Support (# samples)', fontweight='bold', fontsize=12)
axes[1, 1].set_title('Test Set Distribution', fontweight='bold', fontsize=14)
axes[1, 1].grid(axis='y', alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'deliverable3_v4_per_class_metrics.png'),
            dpi=300, bbox_inches='tight')
print(f"  Saved: deliverable3_v4_per_class_metrics.png")
plt.close()

# ========== Ensemble Component Analysis ==========
print(f"Generating ensemble component analysis...")
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# CLIP scores
axes[0, 0].hist(predictions_df[predictions_df['correct']]['clip_score'],
                bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
axes[0, 0].hist(predictions_df[~predictions_df['correct']]['clip_score'],
                bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
axes[0, 0].set_xlabel('CLIP Score', fontweight='bold', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontweight='bold', fontsize=12)
axes[0, 0].set_title('CLIP Score Distribution', fontweight='bold', fontsize=14)
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(alpha=0.3)

# Keyword scores
axes[0, 1].hist(predictions_df[predictions_df['correct']]['keyword_score'],
                bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
axes[0, 1].hist(predictions_df[~predictions_df['correct']]['keyword_score'],
                bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
axes[0, 1].set_xlabel('Keyword Score', fontweight='bold', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontweight='bold', fontsize=12)
axes[0, 1].set_title('Keyword Score Distribution',
                     fontweight='bold', fontsize=14)
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(alpha=0.3)

# Path scores
axes[1, 0].hist(predictions_df[predictions_df['correct']]['path_score'],
                bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
axes[1, 0].hist(predictions_df[~predictions_df['correct']]['path_score'],
                bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
axes[1, 0].set_xlabel('Path Score', fontweight='bold', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontweight='bold', fontsize=12)
axes[1, 0].set_title('Path Score Distribution', fontweight='bold', fontsize=14)
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(alpha=0.3)

# Final confidence
axes[1, 1].hist(predictions_df[predictions_df['correct']]['confidence'],
                bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
axes[1, 1].hist(predictions_df[~predictions_df['correct']]['confidence'],
                bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
axes[1, 1].set_xlabel('Final Confidence', fontweight='bold', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontweight='bold', fontsize=12)
axes[1, 1].set_title('Final Confidence Distribution',
                     fontweight='bold', fontsize=14)
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'deliverable3_v4_ensemble_components.png'),
            dpi=300, bbox_inches='tight')
print(f"  Saved: deliverable3_v4_ensemble_components.png")
plt.close()

# ========== System Evolution ==========
print(f"Generating system evolution comparison...")
evolution_data = {
    'Version': ['D2\nResNet50', 'D3 v1\nCLIP', 'D3 v2\nCLIP+KW', 'D3 v4\nEnsemble'],
    'Accuracy': [0.566, 0.620, 0.680, accuracy],
    'Precision': [0.572, 0.635, 0.695, precision.mean()],
    'Recall': [0.566, 0.620, 0.680, recall.mean()],
    'F1-Score': [0.568, 0.625, 0.685, f1.mean()]
}

evolution_df = pd.DataFrame(evolution_data)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Line plot
x = np.arange(len(evolution_df))
axes[0].plot(x, evolution_df['Accuracy'], marker='o', linewidth=3,
             markersize=10, label='Accuracy', color='steelblue')
axes[0].plot(x, evolution_df['Precision'], marker='s', linewidth=3,
             markersize=10, label='Precision', color='coral')
axes[0].plot(x, evolution_df['Recall'], marker='^', linewidth=3,
             markersize=10, label='Recall', color='seagreen')
axes[0].plot(x, evolution_df['F1-Score'], marker='D', linewidth=3,
             markersize=10, label='F1-Score', color='mediumpurple')

axes[0].set_xticks(x)
axes[0].set_xticklabels(evolution_df['Version'], fontsize=10)
axes[0].set_ylabel('Score', fontweight='bold', fontsize=13)
axes[0].set_title('System Performance Evolution',
                  fontweight='bold', fontsize=15)
axes[0].set_ylim([0.5, 0.85])
axes[0].legend(fontsize=12, loc='lower right')
axes[0].grid(alpha=0.3)

# Add value labels
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    for i, val in enumerate(evolution_df[metric]):
        axes[0].text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')

# Bar chart
bar_width = 0.2
x_pos = np.arange(len(evolution_df))

axes[1].bar(x_pos - 1.5*bar_width, evolution_df['Accuracy'], bar_width,
            label='Accuracy', color='steelblue', alpha=0.8)
axes[1].bar(x_pos - 0.5*bar_width, evolution_df['Precision'], bar_width,
            label='Precision', color='coral', alpha=0.8)
axes[1].bar(x_pos + 0.5*bar_width, evolution_df['Recall'], bar_width,
            label='Recall', color='seagreen', alpha=0.8)
axes[1].bar(x_pos + 1.5*bar_width, evolution_df['F1-Score'], bar_width,
            label='F1-Score', color='mediumpurple', alpha=0.8)

axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(evolution_df['Version'], fontsize=10)
axes[1].set_ylabel('Score', fontweight='bold', fontsize=13)
axes[1].set_title('Performance Comparison', fontweight='bold', fontsize=15)
axes[1].set_ylim([0, 0.9])
axes[1].legend(fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'deliverable3_v4_evolution.png'),
            dpi=300, bbox_inches='tight')
print(f"  Saved: deliverable3_v4_evolution.png")
plt.close()

# ========== Save Report ==========
print(f"\nSaving comprehensive metrics report...")

report = {
    "deliverable": "3.0 v4.0",
    "model": "Ensemble Classifier (CLIP ViT-B/32 + Keyword + Path + Smart Validator)",
    "date": "January 2025",
    "test_accuracy": float(accuracy),
    "per_class_metrics": {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i])
        } for i in range(num_classes)
    },
    "macro_averages": {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1_score": float(f1.mean())
    },
    "weighted_averages": {
        "precision": float(weighted_p),
        "recall": float(weighted_r),
        "f1_score": float(weighted_f1)
    },
    "ensemble_analysis": {
        "clip_contribution": "95% (primary visual understanding)",
        "keyword_contribution": "3% (category disambiguation)",
        "path_contribution": "2% (file naming patterns)",
        "avg_clip_score_correct": float(predictions_df[predictions_df['correct']]['clip_score'].mean()),
        "avg_clip_score_error": float(predictions_df[~predictions_df['correct']]['clip_score'].mean())
    },
    "error_analysis": {
        "total_errors": int((~predictions_df['correct']).sum()),
        "error_rate": float((~predictions_df['correct']).sum() / len(predictions_df)),
        "avg_error_confidence": float(predictions_df[~predictions_df['correct']]['confidence'].mean()),
        "avg_correct_confidence": float(predictions_df[predictions_df['correct']]['confidence'].mean())
    },
    "improvements_from_baseline": {
        "baseline_model": "ResNet50 (Deliverable 2)",
        "baseline_accuracy": 0.566,
        "current_accuracy": float(accuracy),
        "absolute_gain": float(accuracy - 0.566),
        "relative_gain": f"+{(accuracy - 0.566)*100:.2f}%"
    }
}

output_path = os.path.join(RESULTS_DIR, "deliverable3_v4_metrics.json")
with open(output_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"  Saved: deliverable3_v4_metrics.json")

print(f"\n{'='*70}")
print("EVALUATION COMPLETE")
print(f"{'='*70}\n")
print(f"Generated artifacts:")
print(f"  - deliverable3_v4_confusion_matrix.png")
print(f"  - deliverable3_v4_per_class_metrics.png")
print(f"  - deliverable3_v4_ensemble_components.png")
print(f"  - deliverable3_v4_evolution.png")
print(f"  - deliverable3_v4_metrics.json")
print(f"\nAll visualizations saved to: {RESULTS_DIR}/")
print(f"\nReady for IEEE report writing!")
