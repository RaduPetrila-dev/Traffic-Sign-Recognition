"""Interactive Gradio demo: drag in a sign, see predictions + Grad-CAM.

Loads the trained ensemble and exposes a web UI that returns the top-k
class probabilities alongside a Grad-CAM overlay showing where the model
looked. Designed to run locally or on Hugging Face Spaces.

Usage:
    pip install gradio
    python scripts/app.py
"""

import os
import sys

import gradio as gr
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEVICE, MODEL_DIR, NUM_ENSEMBLE  # noqa: E402
from src.data import get_val_transforms  # noqa: E402
from src.gradcam import generate_gradcam_for_image  # noqa: E402
from src.labels import get_sign_name  # noqa: E402
from src.model import create_model, ensemble_predict  # noqa: E402

TRANSFORM = get_val_transforms()


def load_ensemble():
    """Load the trained ensemble; returns None if checkpoints are missing."""
    models_list = []
    for i in range(NUM_ENSEMBLE):
        path = os.path.join(MODEL_DIR, f"traffic_sign_model_{i}.pth")
        if not os.path.exists(path):
            return None
        model = create_model(compile_model=False)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        models_list.append(model)
    return models_list


ENSEMBLE = load_ensemble()


def predict(image: Image.Image, top_k: int = 3):
    """Run ensemble inference and Grad-CAM on an uploaded image."""
    if image is None:
        return {}, None
    if ENSEMBLE is None:
        raise gr.Error(
            f"No checkpoints found in {MODEL_DIR}. Train the model "
            "(`make train`) or download released weights first."
        )

    image = image.convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = ensemble_predict(ENSEMBLE, tensor).squeeze(0)

    k = min(top_k, probs.numel())
    top_probs, top_idx = torch.topk(probs, k)
    labels = {
        get_sign_name(int(i)): float(p)
        for p, i in zip(top_probs.cpu(), top_idx.cpu())
    }

    # Grad-CAM overlay from the first ensemble member for interpretability.
    overlay, _, _, _ = generate_gradcam_for_image(ENSEMBLE[0], image, tensor)
    return labels, Image.fromarray(np.uint8(overlay))


def build_demo():
    with gr.Blocks(title="Traffic Sign Recognition") as demo:
        gr.Markdown(
            "# Traffic Sign Recognition\n"
            "Upload a cropped traffic sign. The model returns the top "
            "predictions and a Grad-CAM heatmap showing where it looked."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Traffic sign")
                top_k = gr.Slider(1, 5, value=3, step=1, label="Top-k")
                btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                out_labels = gr.Label(num_top_classes=5, label="Predictions")
                out_cam = gr.Image(label="Grad-CAM overlay")
        btn.click(predict, inputs=[inp, top_k], outputs=[out_labels, out_cam])
    return demo


if __name__ == "__main__":
    build_demo().launch()
