"""Model definition and ensemble logic.

Uses ResNet18 with pretrained ImageNet weights via the modern
torchvision.models.ResNet18_Weights API (replaces deprecated pretrained=True).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.config import DEVICE, NUM_CLASSES, NUM_ENSEMBLE


def create_model() -> nn.Module:
    """Create a ResNet18 model fine-tuned for traffic sign classification.

    Architecture:
        ResNet18 backbone (early layers frozen) -> Global Avg Pool
        -> Dropout(0.5) -> Linear(512 -> NUM_CLASSES)
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze early layers: only fine-tune the last few residual blocks
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, NUM_CLASSES),
    )
    return model.to(DEVICE)


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
