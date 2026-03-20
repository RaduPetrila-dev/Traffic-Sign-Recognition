# Traffic Sign Recognition for Autonomous Systems

A modular deep learning pipeline for traffic sign detection and classification, built with safety-critical deployment in mind. Uses a ResNet-18 ensemble with OpenCV preprocessing, Grad-CAM interpretability, ONNX export for edge deployment, and an end-to-end YOLO detection pipeline.

![CI](https://github.com/RaduPetrila-dev/Traffic-Sign-Recognition/actions/workflows/ci.yml/badge.svg)

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 96.8% |
| Training Time | ~15 min (GPU) |
| GPU Inference | ~50 ms / batch of 64 |
| CPU Inference | ~180 ms / batch of 64 |
| Model Size | 44.7 MB per model |
| Unit Tests | 40 |

## Project Structure

```
traffic-sign-recognition/
├── main.py                          # Train, evaluate, benchmark
├── Makefile                         # All common commands
├── Dockerfile                       # Reproducible environment
├── requirements.txt
├── .github/workflows/ci.yml         # GitHub Actions CI
│
├── src/
│   ├── config.py                    # Centralised hyperparameters
│   ├── data.py                      # Data loading, OpenCV CLAHE, augmentation
│   ├── model.py                     # ResNet-18 architecture, ensemble logic
│   ├── train.py                     # Training loop, early stopping
│   ├── evaluate.py                  # Evaluation, plots, benchmarking
│   ├── gradcam.py                   # Grad-CAM heatmap generation
│   ├── export.py                    # ONNX export for edge deployment
│   ├── detect.py                    # YOLOv8 detection + classification pipeline
│   └── labels.py                    # GTSRB class name lookup (all 43 classes)
│
├── scripts/
│   ├── infer.py                     # Single-image inference CLI
│   ├── gradcam_viz.py               # Grad-CAM visualisation CLI
│   ├── export_onnx.py               # ONNX export CLI
│   ├── detect_and_classify.py       # Detection + classification CLI
│   └── demo_webcam.py               # Real-time webcam demo
│
├── tests/
│   └── test_pipeline.py             # 40 unit tests
│
├── checkpoints/                     # Saved model weights (generated)
├── outputs/                         # Plots and visualisations (generated)
└── exports/                         # ONNX models (generated)
```

## Quick Start

```bash
pip install -r requirements.txt

# Download GTSRB from Kaggle:
# https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
# Extract to ./gtsrb-german-traffic-sign/Train/

make train           # Train ensemble (3 models)
make test            # Run 40 unit tests
make lint            # Lint all source files
```

## Features

### OpenCV CLAHE Preprocessing

Every image passes through Contrast Limited Adaptive Histogram Equalisation (CLAHE) on the L channel of LAB colour space before augmentation. This normalises contrast across lighting conditions, simulating the variability of real-world capture from moving vehicles. Implemented as a composable torchvision transform in `src/data.py`.

### Grad-CAM Interpretability

Generates heatmaps showing which image regions the model attends to when classifying. Critical for safety-critical systems where understanding failure modes matters as much as accuracy.

```bash
make gradcam IMG=path/to/sign.png

# Or with options:
python scripts/gradcam_viz.py path/to/sign.png --model-idx 0 --target-class 14
```

Produces a side-by-side visualisation: original | heatmap | overlay with predicted class and confidence.

### ONNX Export

Export trained models to ONNX format for deployment on TensorRT, OpenVINO, ONNX Runtime, or mobile runtimes without requiring PyTorch at inference time.

```bash
make export

# Or export a single model:
python scripts/export_onnx.py --model-idx 0
```

### End-to-End Detection + Classification

Uses YOLOv8-nano for localising traffic signs in full scene images, then feeds each crop into the ResNet-18 ensemble for fine-grained classification. Moves the system from "classify cropped patches" to "detect and classify in the wild."

```bash
# Requires: pip install ultralytics
make detect IMG=path/to/scene.jpg

# Or with options:
python scripts/detect_and_classify.py scene.jpg --detector yolov8s.pt --save
```

### Real-Time Webcam Demo

Opens a webcam feed, runs ensemble inference on each frame, and overlays predictions with confidence and FPS counter.

```bash
make demo

# Or with options:
python scripts/demo_webcam.py --camera 0 --threshold 0.8
```

### Single-Image Inference

```bash
make infer IMG=test_image.png

# Output:
# Predictions for: test_image.png
# -------------------------------------------------------
#   Stop                                          97.82%
#   No entry                                       1.43%
#   Yield                                          0.31%
```

## Architecture

**Classifier:** ResNet-18 pretrained on ImageNet (via `ResNet18_Weights.DEFAULT`), early layers frozen, custom head:

```
Input (224x224x3) -> ResNet-18 backbone -> Global Avg Pool
    -> Dropout(0.5) -> Linear(512 -> 43)
```

**Ensemble:** 3 models with different random initialisations. Predictions averaged via softmax probabilities.

**Preprocessing:** OpenCV CLAHE on LAB L-channel, then torchvision augmentation (rotation, colour jitter, Gaussian blur, affine transforms, Gaussian noise injection).

**Class imbalance:** WeightedRandomSampler with inverse frequency weighting.

**Training:** Adam optimiser, ReduceLROnPlateau scheduling, early stopping (patience=5), best-checkpoint saving.

## Safety Analysis

The evaluation module ranks the top misclassification pairs by frequency. Knowing that the model confuses class 1 (30 km/h) with class 2 (50 km/h) is more actionable than knowing overall accuracy is 96.8%. Per-class accuracy plots highlight classes below 90% in red for visual triage.

Grad-CAM heatmaps provide a second layer of safety analysis: if the model classifies a stop sign correctly but attends to the background rather than the sign itself, that prediction is unreliable even though the accuracy metric looks fine.

## Testing

40 unit tests covering:

- Model output shape, softmax validity, dropout presence
- Frozen vs trainable layer verification
- Ensemble member independence and probability constraints
- OpenCV CLAHE (type, size, mode preservation, varying clip limits)
- Transform determinism and output shapes
- Gaussian noise clamping and zero-std identity
- Grad-CAM heatmap shape, range, overlay dimensions and dtype
- Grad-CAM convenience function end-to-end
- ONNX export file creation and size validation
- Label completeness (all 43 classes) and lookup correctness
- Config consistency with model architecture

```bash
make test
```

## Docker

```bash
make docker-build    # Build image
make docker-test     # Run tests in container

# Run inference in container:
docker run --rm -v $(pwd)/checkpoints:/app/checkpoints \
    traffic-sign-recognition python scripts/infer.py /app/test.png
```

## Configuration

All hyperparameters in `src/config.py`:

```python
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
PATIENCE = 5
NUM_ENSEMBLE = 3
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 43
GRADCAM_TARGET_LAYER = "layer4"
```

## Dataset

[German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html): 43 classes, 39,209 images from German roads.

> J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel. "The German Traffic Sign Recognition Benchmark: A multi-class classification competition." IEEE IJCNN, 2011.

## License

MIT

## Author

**Radu Petrila** — [@RaduPetrila-dev](https://github.com/RaduPetrila-dev) — [LinkedIn](https://www.linkedin.com/in/radu-petrila-96762b2b2) — sebastianpetrila8@gmail.com
