"""Run inference on a single image using the trained ensemble.

Usage:
    python scripts/infer.py path/to/image.png
    python scripts/infer.py path/to/image.png --top-k 5
"""

import argparse
import os
import sys

import cv2
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEVICE, MODEL_DIR, NUM_ENSEMBLE  # noqa: E402
from src.data import get_val_transforms  # noqa: E402
from src.labels import get_sign_name  # noqa: E402
from src.model import create_model, ensemble_predict  # noqa: E402


def load_ensemble():
    """Load trained ensemble from checkpoints."""
    models_list = []
    for i in range(NUM_ENSEMBLE):
        path = os.path.join(MODEL_DIR, f"traffic_sign_model_{i}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        model = create_model()
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        models_list.append(model)
    return models_list


def preprocess_image(image_path: str) -> tuple:
    """Load and preprocess a single image. Returns (tensor, pil_image)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = get_val_transforms()(pil_img).unsqueeze(0).to(DEVICE)
    return tensor, pil_img


def infer(image_path: str, top_k: int = 3):
    """Run ensemble inference and print top-k predictions."""
    models_list = load_ensemble()
    image_tensor, _ = preprocess_image(image_path)

    with torch.no_grad():
        probs = ensemble_predict(models_list, image_tensor)

    top_probs, top_classes = torch.topk(probs, top_k, dim=1)
    top_probs = top_probs.squeeze().cpu().numpy()
    top_classes = top_classes.squeeze().cpu().numpy()

    print(f"\nPredictions for: {image_path}")
    print("-" * 55)
    for prob, cls in zip(top_probs, top_classes):
        print(f"  {get_sign_name(int(cls)):45s} {prob * 100:6.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic sign inference")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    infer(args.image, top_k=args.top_k)
