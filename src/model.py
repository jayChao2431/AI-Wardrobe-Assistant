# All comments in English.
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(in_feats, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits, feats


def freeze_backbone(model: nn.Module, unfreeze_last_n: int = 0):
    """
    Freeze backbone layers for transfer learning.

    Args:
        model: ResNet50Classifier instance
        unfreeze_last_n: Number of layer blocks to unfreeze (0-4)
                        0 = Freeze entire backbone (recommended for small datasets)
                        1 = Unfreeze layer4 only
                        2 = Unfreeze layer3 + layer4
                        etc.

    Note: For small datasets (<1000 images), keeping unfreeze_last_n=0 is recommended
          to prevent overfitting. Only the classifier head will be trained.
    """
    # Freeze all backbone parameters first
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Unfreeze specified layers if requested
    layers = [model.backbone.layer4, model.backbone.layer3,
              model.backbone.layer2, model.backbone.layer1]

    for i in range(min(unfreeze_last_n, len(layers))):
        for p in layers[i].parameters():
            p.requires_grad = True


def get_parameter_groups(model: nn.Module, lr_backbone: float = 1e-4, lr_classifier: float = 1e-3):
    """
    Create parameter groups with differential learning rates.

    Args:
        model: ResNet50Classifier instance
        lr_backbone: Learning rate for backbone (lower for pretrained features)
        lr_classifier: Learning rate for classifier head (higher for new layers)

    Returns:
        List of parameter groups for optimizer

    Example usage:
        param_groups = get_parameter_groups(model, lr_backbone=1e-4, lr_classifier=1e-3)
        optimizer = torch.optim.Adam(param_groups)
    """
    # Collect trainable parameters from backbone
    backbone_params = [p for p in model.backbone.parameters()
                       if p.requires_grad]

    # Classifier parameters
    classifier_params = list(model.classifier.parameters())

    param_groups = []

    if len(backbone_params) > 0:
        param_groups.append({
            'params': backbone_params,
            'lr': lr_backbone,
            'name': 'backbone'
        })

    param_groups.append({
        'params': classifier_params,
        'lr': lr_classifier,
        'name': 'classifier'
    })

    return param_groups
