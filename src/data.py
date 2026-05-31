"""Data loading, preprocessing, and augmentation pipeline.

Uses OpenCV for image preprocessing (colour space conversion, histogram
equalisation, adaptive filtering) before the standard torchvision transforms.
"""

import os
import random
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms

from src.config import (
    BATCH_SIZE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, DATA_DIR,
    IMG_HEIGHT, IMG_WIDTH, IMAGENET_MEAN, IMAGENET_STD, NUM_WORKERS, SEED,
)


class OpenCVPreprocess:
    """Apply OpenCV preprocessing before torchvision transforms.

    Converts to LAB colour space, applies CLAHE (Contrast Limited Adaptive
    Histogram Equalisation) to the L channel, and converts back to RGB.
    This improves robustness under varying lighting conditions, which is
    common in real-world traffic sign images captured from moving vehicles.
    """

    def __init__(
        self,
        clip_limit: float = CLAHE_CLIP_LIMIT,
        tile_grid_size: tuple = CLAHE_TILE_GRID_SIZE,
    ):
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
        OpenCVPreprocess(),
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
        OpenCVPreprocess(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class TransformSubset(Dataset):
    """A subset of a dataset that applies its own transform.

    Unlike reassigning ``Subset.dataset``, this keeps each split bound to an
    explicit transform without re-scanning the image directory or mutating a
    shared dataset object.
    """

    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, class_idx)
        self.transform = transform
        self.targets = [label for _, label in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def _track_id(path: str) -> str:
    """Extract the GTSRB track id from an image filename.

    GTSRB images are named ``<track>_<frame>.<ext>`` (e.g. ``00012_00003.png``);
    every frame in a track is the same physical sign. Returns the filename stem
    if the convention is not present, so each image becomes its own group.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0] if "_" in stem else stem


def track_aware_split(samples, ratios=(0.7, 0.15, 0.15), seed: int = SEED):
    """Split samples into train/val/test without leaking GTSRB tracks.

    Frames from the same physical sign (a "track") are near-duplicates. A naive
    per-image random split scatters them across splits and inflates test
    accuracy. We instead group by (class, track) and split whole groups, so no
    sign appears in more than one split.

    Returns:
        tuple of three sample lists (train, val, test).
    """
    groups = defaultdict(list)
    for path, label in samples:
        groups[(label, _track_id(path))].append((path, label))

    keys = sorted(groups.keys())
    random.Random(seed).shuffle(keys)

    n_train = int(ratios[0] * len(keys))
    n_val = int(ratios[1] * len(keys))
    splits = {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }
    return tuple(
        [sample for key in splits[name] for sample in groups[key]]
        for name in ("train", "val", "test")
    )


def build_weighted_sampler(samples):
    """Build a WeightedRandomSampler to handle class imbalance.

    GTSRB has severe imbalance (some classes have 10x more samples).
    Inverse frequency weighting ensures the model sees all classes equally.
    """
    labels = [label for _, label in samples]
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(data_dir: str = DATA_DIR):
    """Load GTSRB data with a track-aware 70/15/15 train/val/test split.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. Download from "
            "https://www.kaggle.com/datasets/"
            "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
        )

    base = datasets.ImageFolder(root=data_dir)
    train_samples, val_samples, test_samples = track_aware_split(base.samples)

    train_dataset = TransformSubset(train_samples, get_train_transforms())
    val_dataset = TransformSubset(val_samples, get_val_transforms())
    test_dataset = TransformSubset(test_samples, get_val_transforms())

    sampler = build_weighted_sampler(train_samples)

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": NUM_WORKERS > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    print(
        f"Dataset sizes - Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader, base.classes
