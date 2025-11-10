# All comments in English.
from typing import Tuple, Dict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transforms(img_size: int = 224):
    train_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    eval_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return train_t, eval_t

def get_dataloaders(root: str, batch_size: int = 16, num_workers: int = 2, img_size: int = 224):
    train_t, eval_t = get_transforms(img_size)
    train_ds = datasets.ImageFolder(f"{root}/train", transform=train_t)
    val_ds   = datasets.ImageFolder(f"{root}/val", transform=eval_t)
    test_ds  = datasets.ImageFolder(f"{root}/test", transform=eval_t)

    idx_to_class = {i:c for c,i in train_ds.class_to_idx.items()}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, idx_to_class
