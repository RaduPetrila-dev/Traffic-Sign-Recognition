"""Run inference on a single image using the trained ensemble.

Usage:
    python scripts/infer.py path/to/image.png
    python scripts/infer.py path/to/image.png --top-k 5
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEVICE, MODEL_DIR, NUM_CLASSES, NUM_ENSEMBLE
from src.data import get_val_transforms
from src.model import create_model, ensemble_predict

# GTSRB class names (subset of common ones for display)
SIGN_NAMES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)", 3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)", 9: "No passing",
    10: "No passing (vehicles > 3.5t)", 11: "Right-of-way at intersection",
    12: "Priority road", 13: "Yield", 14: "Stop",
    15: "No vehicles", 16: "Vehicles > 3.5t prohibited",
    17: "No entry", 18: "General caution", 19: "Dangerous curve left",
    20: "Dangerous curve right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road",
    24: "Road narrows on the right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow",
    31: "Wild animals crossing", 32: "End of all restrictions",
    33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead only", 36: "Go straight or right",
    37: "Go straight or left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout mandatory", 41: "End of no passing",
    42: "End of no passing (vehicles > 3.5t)",
}


def load_ensemble():
    """Load trained ensemble from checkpoints."""
    models_list = []
    for i in range(NUM_ENSEMBLE):
        path = os.path.join(MODEL_DIR, f"traffic_sign_model_{i}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}. Run training first."
            )
        model = create_model()
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        models_list.append(model)
    return models_list


def preprocess_image(image_path: str) -> torch.Tensor:
    """Load and preprocess a single image for inference.

    Uses OpenCV to read the image, then applies the same validation
    transforms used during training (CLAHE + resize + normalise).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    transform = get_val_transforms()
    tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension
    return tensor.to(DEVICE)


def infer(image_path: str, top_k: int = 3):
    """Run inference on a single image and print top-k predictions."""
    models_list = load_ensemble()
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        probs = ensemble_predict(models_list, image_tensor)

    top_probs, top_classes = torch.topk(probs, top_k, dim=1)
    top_probs = top_probs.squeeze().cpu().numpy()
    top_classes = top_classes.squeeze().cpu().numpy()

    print(f"\nPredictions for: {image_path}")
    print("-" * 50)
    for prob, cls in zip(top_probs, top_classes):
        name = SIGN_NAMES.get(cls, f"Class {cls}")
        print(f"  {name:45s} {prob * 100:5.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic sign inference")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions")
    args = parser.parse_args()

    infer(args.image, top_k=args.top_k)
