# Traffic Sign Recognition for Autonomous Systems

A deep learning pipeline for traffic sign classification on the [GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html) dataset, built with safety-critical deployment in mind. Uses a ResNet-18 ensemble with OpenCV preprocessing, comprehensive test coverage, and inference benchmarking.

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 96.8% |
| Training Time | ~15 min (GPU) |
| GPU Inference | ~50ms / batch of 64 |
| CPU Inference | ~180ms / batch of 64 |
| Model Size | 44.7 MB per model |
| Unit Tests | 20 |

## Why This Matters

Traffic sign recognition is safety-critical. Misclassifying a stop sign as a yield sign has consequences that a 1% accuracy drop on ImageNet does not. This pipeline addresses that through:

- **Per-class safety analysis** identifying the most dangerous misclassification pairs
- **Sensor degradation simulation** (blur, noise, lighting variation) to test robustness
- **OpenCV CLAHE preprocessing** for consistent performance across lighting conditions
- **Ensemble averaging** across 3 models to reduce variance on edge cases
- **Inference benchmarking** to verify deployment feasibility on GPU and CPU

## Project Structure

```
traffic-sign-recognition/
├── main.py                  # Entry point: train, evaluate, benchmark
├── Makefile                 # train / test / infer / lint / clean
├── requirements.txt
├── src/
│   ├── config.py            # Centralised hyperparameters and paths
│   ├── data.py              # Data loading, OpenCV preprocessing, augmentation
│   ├── model.py             # ResNet-18 architecture and ensemble logic
│   ├── train.py             # Training loop with early stopping
│   └── evaluate.py          # Evaluation, visualisation, benchmarking
├── scripts/
│   └── infer.py             # Single-image inference CLI
├── tests/
│   └── test_pipeline.py     # 20 unit tests (model, transforms, ensemble)
├── checkpoints/             # Saved model weights (generated)
└── outputs/                 # Plots and evaluation outputs (generated)
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Download GTSRB dataset from Kaggle:
# https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
# Extract to ./gtsrb-german-traffic-sign/Train/

# Train ensemble
make train          # or: python main.py

# Run tests
make test           # or: python -m pytest tests/ -v

# Inference on a single image
python scripts/infer.py path/to/sign.png --top-k 5

# Lint
make lint
```

## Architecture

**Model:** ResNet-18 pretrained on ImageNet (via `ResNet18_Weights.DEFAULT`), with early layers frozen and a custom classification head:

```
Input (224x224x3) -> ResNet-18 backbone (frozen early layers)
    -> Global Average Pooling -> Dropout(0.5) -> Linear(512 -> 43)
```

**Ensemble:** 3 models trained with different random initialisations. Final prediction averages softmax probabilities across all members.

**Preprocessing (OpenCV):** CLAHE (Contrast Limited Adaptive Histogram Equalisation) applied to the L channel of LAB colour space before augmentation. This normalises contrast across varying lighting conditions, simulating real-world capture from moving vehicles.

**Augmentation pipeline:**
- Random rotation (15 degrees)
- Colour jitter (brightness, contrast, saturation)
- Gaussian blur (simulating camera defocus)
- Affine transforms (translation, scale)
- Gaussian noise injection (simulating sensor noise)

**Class imbalance:** WeightedRandomSampler with inverse frequency weighting.

## Testing

20 unit tests covering:

- Model output shape and softmax validity
- Frozen vs trainable layer verification
- Ensemble prediction shape and probability constraints
- OpenCV preprocessing (type, size, colour mode preservation)
- Transform output shapes for training and validation
- Gaussian noise clamping to valid range
- Validation transform determinism (no random augmentation leaking)
- Configuration consistency checks

```bash
python -m pytest tests/ -v
```

## Safety Analysis

The evaluation module prints the top 10 misclassification pairs, ranked by frequency. This is the most operationally relevant metric for deployed vision systems: knowing that your model confuses class 1 (30 km/h) with class 2 (50 km/h) is more actionable than knowing overall accuracy is 96.8%.

Per-class accuracy plots highlight classes below 90% accuracy in red for visual triage.

## Single-Image Inference

```bash
python scripts/infer.py test_image.png --top-k 3

# Output:
# Predictions for: test_image.png
# --------------------------------------------------
#   Stop                                          97.82%
#   No entry                                       1.43%
#   Yield                                          0.31%
```

Uses the same OpenCV CLAHE preprocessing as training for consistency.

## Configuration

All hyperparameters are centralised in `src/config.py`:

```python
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
PATIENCE = 5          # Early stopping
NUM_ENSEMBLE = 3
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 43
```

## Dataset

[German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html): 43 classes, 39,209 training images from German roads with varying lighting, weather, and occlusion.

> J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. "The German Traffic Sign Recognition Benchmark: A multi-class classification competition." IEEE IJCNN, 2011.

## License

MIT

## Author

**Radu Petrila**
- GitHub: [@RaduPetrila-dev](https://github.com/RaduPetrila-dev)
- LinkedIn: [Radu Petrila](https://www.linkedin.com/in/radu-petrila-96762b2b2)
- Email: sebastianpetrila8@gmail.com
