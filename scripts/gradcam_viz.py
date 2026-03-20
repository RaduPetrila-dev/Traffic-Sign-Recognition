"""Generate Grad-CAM heatmap overlays for traffic sign images.

Produces a side-by-side visualisation: original image | heatmap | overlay,
saved to the outputs directory. Useful for debugging misclassifications
and understanding what the model attends to.

Usage:
    python scripts/gradcam_viz.py path/to/image.png
    python scripts/gradcam_viz.py path/to/image.png --model-idx 0
    python scripts/gradcam_viz.py path/to/image.png --target-class 14
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEVICE, IMG_HEIGHT, IMG_WIDTH, MODEL_DIR, OUTPUT_DIR  # noqa: E402
from src.data import get_val_transforms  # noqa: E402
from src.gradcam import GradCAM  # noqa: E402
from src.labels import get_sign_name  # noqa: E402
from src.model import create_model  # noqa: E402


def run_gradcam(image_path: str, model_idx: int = 0, target_class: int = None):
    """Generate and save a Grad-CAM visualisation."""
    # Load model
    checkpoint = os.path.join(MODEL_DIR, f"traffic_sign_model_{model_idx}.pth")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = create_model()
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    # Load and preprocess image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    transform = get_val_transforms()
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = probs.max(dim=1)
    pred_class = predicted.item()
    pred_conf = confidence.item()

    if target_class is None:
        target_class = pred_class

    # Generate Grad-CAM
    cam = GradCAM(model)
    heatmap = cam.generate(input_tensor, target_class)

    # Prepare original image at model input size
    original_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    overlay = cam.overlay(original_resized, heatmap, alpha=0.5)

    # Create side-by-side figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_resized)
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"Grad-CAM (class {target_class})", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    pred_name = get_sign_name(pred_class)
    axes[2].set_title(f"Overlay: {pred_name} ({pred_conf:.1%})", fontsize=13)
    axes[2].axis("off")

    plt.tight_layout()

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"gradcam_{basename}.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    print(f"Prediction: {pred_name} ({pred_conf:.1%})")
    print(f"Grad-CAM target class: {get_sign_name(target_class)} ({target_class})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM visualisation")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model-idx", type=int, default=0,
                        help="Which ensemble member to visualise (0-2)")
    parser.add_argument("--target-class", type=int, default=None,
                        help="Class to visualise (default: predicted class)")
    args = parser.parse_args()
    run_gradcam(args.image, model_idx=args.model_idx, target_class=args.target_class)
