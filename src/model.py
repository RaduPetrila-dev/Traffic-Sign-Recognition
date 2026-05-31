"""Model definition and ensemble logic.

Uses ResNet18 with pretrained ImageNet weights via the modern
torchvision.models.ResNet18_Weights API (replaces deprecated pretrained=True).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.config import DEVICE, NUM_CLASSES, NUM_ENSEMBLE, USE_COMPILE

# Backbone modules left trainable; everything earlier is frozen. Naming
# modules (rather than slicing the parameter list) makes the fine-tuning
# scope explicit and robust to changes in torchvision's parameter ordering.
TRAINABLE_BACKBONE_LAYERS = ("layer4",)


def create_model(compile_model: bool = USE_COMPILE) -> nn.Module:
    """Create a ResNet18 model fine-tuned for traffic sign classification.

    Architecture:
        ResNet18 backbone (early layers frozen) -> Global Avg Pool
        -> Dropout(0.5) -> Linear(512 -> NUM_CLASSES)

    The last residual block (``layer4``) and the new classifier head are
    trainable; all earlier layers are frozen. ``torch.compile`` is applied
    when requested and supported (a no-op fallback on unsupported setups).
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze everything, then unfreeze the chosen backbone tail.
    for param in model.parameters():
        param.requires_grad = False
    for name, module in model.named_children():
        if name in TRAINABLE_BACKBONE_LAYERS:
            for param in module.parameters():
                param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, NUM_CLASSES),
    )
    model = model.to(DEVICE)

    if compile_model and hasattr(torch, "compile") and DEVICE.type == "cuda":
        model = torch.compile(model)

    return model


def create_ensemble() -> list:
    """Create an ensemble of NUM_ENSEMBLE models with different initialisations."""
    return [create_model() for _ in range(NUM_ENSEMBLE)]


def ensemble_predict(models_list: list, images: torch.Tensor) -> torch.Tensor:
    """Average softmax probabilities across all ensemble members.

    Args:
        models_list: list of trained nn.Module models
        images: batch of input images (B, C, H, W)

    Returns:
        Averaged probability tensor (B, NUM_CLASSES)
    """
    ensemble_output = torch.zeros(images.size(0), NUM_CLASSES).to(images.device)
    for model in models_list:
        model.eval()
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        ensemble_output += probs
    ensemble_output /= len(models_list)
    return ensemble_output
