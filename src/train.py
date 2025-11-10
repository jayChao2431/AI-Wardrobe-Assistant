# All comments in English.
import os, json
import torch, torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset_loader import get_dataloaders
from src.model import ResNet50Classifier, freeze_backbone

DATA_ROOT = "data/deepfashion_subset"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss/total, total_correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss/total, total_correct/total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader, idx_to_class = get_dataloaders(DATA_ROOT, batch_size=16, num_workers=2, img_size=224)
    num_classes = len(idx_to_class)

    model = ResNet50Classifier(num_classes=num_classes, pretrained=True)
    freeze_backbone(model, unfreeze_last_n=1)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    epochs = 8

    for ep in range(1, epochs+1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl)

        train_losses.append(tl); val_losses.append(vl)
        train_accs.append(ta);  val_accs.append(va)

        print(f"Epoch {ep}: train_loss={tl:.4f}, val_loss={vl:.4f}, train_acc={ta:.4f}, val_acc={va:.4f}")
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model_best.pth"))
            with open(os.path.join(RESULTS_DIR, "class_to_idx.json"), "w") as f:
                json.dump(idx_to_class, f)

    plt.figure(); plt.plot(train_losses, label='train_loss'); plt.plot(val_losses, label='val_loss'); plt.legend(); plt.title('Loss')
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curve.png')); plt.close()

    plt.figure(); plt.plot(train_accs, label='train_acc'); plt.plot(val_accs, label='val_acc'); plt.legend(); plt.title('Accuracy')
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_curve.png')); plt.close()

if __name__ == "__main__":
    main()
