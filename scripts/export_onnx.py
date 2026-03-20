"""Export trained models to ONNX format for deployment.

Exports all ensemble members or a single model. Optionally validates
the exported file if the onnx package is installed.

Usage:
    python scripts/export_onnx.py                    # Export all 3 models
    python scripts/export_onnx.py --model-idx 0      # Export model 0 only
    python scripts/export_onnx.py --no-validate       # Skip validation
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEVICE, MODEL_DIR, NUM_ENSEMBLE, ONNX_DIR
from src.export import export_to_onnx, validate_onnx
from src.model import create_model


def export(model_idx: int = None, skip_validate: bool = False):
    """Export one or all ensemble members to ONNX."""
    indices = [model_idx] if model_idx is not None else range(NUM_ENSEMBLE)
    os.makedirs(ONNX_DIR, exist_ok=True)

    for i in indices:
        checkpoint = os.path.join(MODEL_DIR, f"traffic_sign_model_{i}.pth")
        if not os.path.exists(checkpoint):
            print(f"Skipping model {i}: checkpoint not found at {checkpoint}")
            continue

        model = create_model()
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        model.eval()

        output_path = os.path.join(ONNX_DIR, f"traffic_sign_model_{i}.onnx")
        export_to_onnx(model, output_path)

        if not skip_validate:
            validate_onnx(output_path)

    print(f"\nAll exports saved to {ONNX_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--model-idx", type=int, default=None,
                        help="Export a single model (0, 1, or 2)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip ONNX validation")
    args = parser.parse_args()
    export(model_idx=args.model_idx, skip_validate=args.no_validate)
