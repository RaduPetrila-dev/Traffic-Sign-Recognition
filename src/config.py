"""Centralised configuration for the traffic sign recognition pipeline."""

import torch

# --- Paths ---
DATA_DIR = "./gtsrb-german-traffic-sign/Train"
MODEL_DIR = "./checkpoints"
OUTPUT_DIR = "./outputs"
ONNX_DIR = "./exports"

# --- Reproducibility ---
SEED = 42

# --- Training ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
PATIENCE = 5
NUM_ENSEMBLE = 3

# DataLoader workers (per-image OpenCV CLAHE is CPU-heavy; parallelise it).
# 0 keeps loading in the main process (safest for debugging / CI).
NUM_WORKERS = 4

# Mixed precision and graph compilation (GPU only; both no-op on CPU).
USE_AMP = True
USE_COMPILE = True

# --- Image ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 43

# --- Preprocessing ---
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# ImageNet normalisation stats (used by pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Grad-CAM ---
GRADCAM_TARGET_LAYER = "layer4"  # Last residual block of ResNet18

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
