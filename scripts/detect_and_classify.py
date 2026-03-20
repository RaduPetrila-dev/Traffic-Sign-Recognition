"""Detect and classify traffic signs in full scene images.

Uses YOLOv8 for localisation, then the trained ResNet-18 ensemble
for fine-grained classification of each detected region.

Usage:
    python scripts/detect_and_classify.py path/to/scene.jpg
    python scripts/detect_and_classify.py path/to/scene.jpg --save
    python scripts/detect_and_classify.py path/to/scene.jpg --detector yolov8s.pt
"""

import argparse
import os
import sys

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEVICE, MODEL_DIR, NUM_ENSEMBLE, OUTPUT_DIR  # noqa: E402
from src.detect import detect_and_classify, load_detector  # noqa: E402
from src.model import create_model  # noqa: E402


def load_classifier_ensemble():
    """Load trained classification ensemble."""
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


def run(image_path: str, detector_model: str = "yolov8n.pt",
        confidence: float = 0.3, save: bool = False):
    """Run detection + classification pipeline on a scene image."""
    print(f"Loading detector ({detector_model})...")
    detector = load_detector(detector_model, confidence)

    print("Loading classifier ensemble...")
    classifier = load_classifier_ensemble()

    print(f"Processing: {image_path}")
    annotated, detections = detect_and_classify(
        image_path, detector, classifier,
    )

    # Print results
    print(f"\nFound {len(detections)} traffic sign(s):")
    print("-" * 60)
    for i, det in enumerate(detections):
        print(f"  [{i+1}] {det.class_name}")
        print(f"      Classification confidence: {det.confidence:.1%}")
        print(f"      Detection confidence:      {det.detection_confidence:.1%}")
        print(f"      Bounding box:              {det.bbox}")

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        basename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"detected_{basename}.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"\nSaved annotated image: {output_path}")
    else:
        cv2.imshow("Detections", annotated)
        print("\nPress any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and classify traffic signs")
    parser.add_argument("image", help="Path to a scene image")
    parser.add_argument("--detector", default="yolov8n.pt",
                        help="YOLOv8 model variant (default: yolov8n.pt)")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Detection confidence threshold")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated image instead of displaying")
    args = parser.parse_args()
    run(args.image, detector_model=args.detector,
        confidence=args.confidence, save=args.save)
