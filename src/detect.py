"""End-to-end detection and classification pipeline.

Uses YOLOv8-nano for localising traffic signs in full scene images,
then feeds each cropped region into the trained ResNet-18 ensemble
for fine-grained classification. This moves the system from
"classify cropped patches" to "detect and classify in the wild."

Requires: pip install ultralytics
"""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from src.config import DEVICE, IMG_HEIGHT, IMG_WIDTH
from src.data import get_val_transforms
from src.labels import get_sign_name
from src.model import ensemble_predict


@dataclass
class Detection:
    """A single detected and classified traffic sign."""
    bbox: tuple  # (x1, y1, x2, y2) pixel coordinates
    class_idx: int
    class_name: str
    confidence: float  # Classification confidence
    detection_confidence: float  # YOLO detection confidence


def load_detector(model_name: str = "yolov8n.pt", confidence: float = 0.3):
    """Load a YOLOv8 detector.

    Uses the nano variant by default for speed. For higher accuracy
    on small signs, use "yolov8s.pt" or "yolov8m.pt".

    Args:
        model_name: YOLOv8 model variant.
        confidence: Minimum detection confidence threshold.

    Returns:
        Configured YOLO model.
    """
    from ultralytics import YOLO
    detector = YOLO(model_name)
    detector.conf = confidence
    return detector


def detect_and_classify(
    image_path: str,
    detector,
    classifier_ensemble: list,
    target_classes: Optional[List[int]] = None,
    min_crop_size: int = 20,
) -> tuple:
    """Detect objects in an image and classify traffic sign crops.

    Args:
        image_path: Path to the input scene image.
        detector: Loaded YOLO model.
        classifier_ensemble: List of trained ResNet-18 models.
        target_classes: YOLO class indices to keep (None = all).
            For COCO, class 11 = "stop sign". Use None to detect
            all objects and attempt classification on each crop.
        min_crop_size: Minimum bounding box dimension in pixels.
            Crops smaller than this are skipped (too small for
            reliable classification).

    Returns:
        tuple: (annotated_image, list_of_Detection)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = get_val_transforms()

    # Run YOLO detection
    results = detector(image_path, verbose=False)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = float(box.conf[0])
            yolo_class = int(box.cls[0])

            # Filter by target classes if specified
            if target_classes is not None and yolo_class not in target_classes:
                continue

            # Skip tiny crops
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < min_crop_size or crop_h < min_crop_size:
                continue

            # Crop and classify
            crop = img_rgb[y1:y2, x1:x2]
            crop_pil = Image.fromarray(crop)
            tensor = transform(crop_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                probs = ensemble_predict(classifier_ensemble, tensor)
                confidence, predicted = probs.max(dim=1)

            class_idx = predicted.item()
            class_conf = confidence.item()

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_idx=class_idx,
                class_name=get_sign_name(class_idx),
                confidence=class_conf,
                detection_confidence=det_conf,
            ))

    # Draw annotations
    annotated = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        label = f"{det.class_name} ({det.confidence:.0%})"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated, detections
