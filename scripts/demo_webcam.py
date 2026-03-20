"""Real-time traffic sign classification from webcam feed.

Opens a webcam, captures frames, runs the trained ensemble on each
frame (centre-cropped), and overlays the prediction with confidence.
Press 'q' to quit.

Usage:
    python scripts/demo_webcam.py
    python scripts/demo_webcam.py --camera 1        # Use camera index 1
    python scripts/demo_webcam.py --threshold 0.8   # Only show predictions above 80%
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
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


def centre_crop_square(frame: np.ndarray) -> np.ndarray:
    """Extract a square crop from the centre of the frame.

    For webcam use, the user holds a traffic sign image in front of
    the camera. The centre crop focuses on the region of interest
    and ignores the background.
    """
    h, w = frame.shape[:2]
    size = min(h, w)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    return frame[y_start:y_start + size, x_start:x_start + size]


def run_demo(camera_idx: int = 0, threshold: float = 0.5):
    """Run the real-time webcam demo."""
    print("Loading ensemble...")
    models_list = load_ensemble()
    transform = get_val_transforms()

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_idx}")

    print(f"Camera opened. Press 'q' to quit. Threshold: {threshold:.0%}")

    fps_samples = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Centre crop and classify
        crop = centre_crop_square(frame)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = ensemble_predict(models_list, tensor)
            confidence, predicted = probs.max(dim=1)

        pred_class = predicted.item()
        pred_conf = confidence.item()
        elapsed = time.time() - start_time

        # Track FPS
        fps_samples.append(1.0 / max(elapsed, 1e-6))
        if len(fps_samples) > 30:
            fps_samples.pop(0)
        fps = np.mean(fps_samples)

        # Draw crop region on frame
        h, w = frame.shape[:2]
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        cv2.rectangle(frame, (x_start, y_start),
                      (x_start + size, y_start + size), (0, 255, 0), 2)

        # Draw prediction if above threshold
        if pred_conf >= threshold:
            label = f"{get_sign_name(pred_class)} ({pred_conf:.0%})"
            colour = (0, 255, 0) if pred_conf > 0.8 else (0, 255, 255)
        else:
            label = f"Low confidence ({pred_conf:.0%})"
            colour = (128, 128, 128)

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Traffic Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Demo ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam demo")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum confidence to display prediction")
    args = parser.parse_args()
    run_demo(camera_idx=args.camera, threshold=args.threshold)
