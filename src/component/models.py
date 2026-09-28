"""Classifier construction.

Both models come from torchvision, either with ImageNet weights or randomly
initialized, and get a new output layer for the dataset's classes. Seed the
global RNG before calling ``build_model``: with the same seed, every condition
of a model starts from identical weights.
"""

import torch.nn as nn
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

MODELS = {
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
}


def weights_name(name: str, pretrained: bool) -> str | None:
    """Return the torchvision weight tag used for ``name``, or None if random."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name][1].name if pretrained else None


def build_model(name: str, pretrained: bool, num_classes: int) -> nn.Module:
    """Build ResNet-50 or ViT-B/16 with a ``num_classes`` output layer.

    Args:
        name: ``"resnet50"`` or ``"vit_b_16"``.
        pretrained: Load ImageNet weights (True) or initialize randomly (False).
        num_classes: Size of the new output layer.
    """
    weights_name(name, pretrained)  # validates the name
    constructor, weights = MODELS[name]
    model = constructor(weights=weights if pretrained else None)
    if name == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model
