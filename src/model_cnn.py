# Placeholder CNN definition for future development.
# You can replace this with a pretrained torchvision model (e.g., resnet18).
import torch
import torch.nn as nn
import torchvision.models as models

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
