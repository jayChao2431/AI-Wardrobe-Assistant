# All comments in English.
import os, json
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.dataset_loader import get_dataloaders
from src.model import ResNet50Classifier

DATA_ROOT = "data/deepfashion_subset"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader, idx_to_class = get_dataloaders(DATA_ROOT, batch_size=32, num_workers=2, img_size=224)
    num_classes = len(idx_to_class)

    model = ResNet50Classifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "model_best.pth"), map_location=device))
    model = model.to(device).eval()

    y_true, y_pred = [], []
    for x, y in test_loader:
        x = x.to(device)
        logits, _ = model(x)
        pred = logits.argmax(1).cpu().numpy()
        y_pred.extend(list(pred)); y_true.extend(list(y.numpy()))

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro')
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump({"accuracy": float(acc), "macro_f1": float(f1)}, f, indent=2)
    print({"accuracy": float(acc), "macro_f1": float(f1)})

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    labels = [idx_to_class[i] for i in range(num_classes)]
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))

if __name__ == "__main__":
    main()
