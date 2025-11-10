# All comments in English.
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(in_feats, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits, feats

def freeze_backbone(model: nn.Module, unfreeze_last_n: int = 1):
    for p in model.backbone.parameters():
        p.requires_grad = False
    if unfreeze_last_n > 0:
        for p in model.backbone.layer4.parameters():
            p.requires_grad = True
