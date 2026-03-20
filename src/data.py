"""Data loading, preprocessing, and augmentation pipeline.

Uses OpenCV for image preprocessing (colour space conversion, histogram
equalisation, adaptive filtering) before the standard torchvision transforms.
"""

import os
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from src.config import (
    BATCH_SIZE, DATA_DIR, IMG_HEIGHT, IMG_WIDTH,
    IMAGENET_MEAN, IMAGENET_STD,
)


class OpenCVPreprocess:
    """Apply OpenCV preprocessing before torchvision transforms.

    Converts to LAB colour space, applies CLAHE (Contrast Limited Adaptive
    Histogram Equalisation) to the L channel, and converts back to RGB.
    This improves robustness under varying lighting conditions, which is
    common in real-world traffic sign images captured from moving vehicles.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        img_array = np.array(img)

        # Convert RGB to LAB colour space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to the L (lightness) channel only
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])

        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)


class GaussianNoise:
    """Add random Gaussian noise to simulate sensor noise."""

    def __init__(self, mean: float = 0.0, std: float = 0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


def get_train_transforms():
    """Training transforms with OpenCV preprocessing and augmentation."""
    return transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        OpenCVPreprocess(clip_limit=2.0),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        GaussianNoise(std=0.02),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """Validation/test transforms with OpenCV preprocessing (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        OpenCVPreprocess(clip_limit=2.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_weighted_sampler(dataset, indices):
    """Build a WeightedRandomSampler to handle class imbalance.

    GTSRB has severe imbalance (some classes have 10x more samples).
    Inverse frequency weighting ensures the model sees all classes equally.
    """
    labels = [dataset.targets[i] for i in indices]
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(data_dir: str = DATA_DIR):
    """Load GTSRB data with 70/15/15 train/val/test split.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. Download from "
            "https://www.kaggle.com/datasets/"
            "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
        )

    full_dataset = datasets.ImageFolder(root=data_dir, transform=get_train_transforms())

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply non-augmented transforms to val/test
    val_dataset.dataset = datasets.ImageFolder(
        root=data_dir, transform=get_val_transforms()
    )
    test_dataset.dataset = datasets.ImageFolder(
        root=data_dir, transform=get_val_transforms()
    )

    sampler = build_weighted_sampler(full_dataset, train_dataset.indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    return train_loader, val_loader, test_loader, full_dataset.classes
