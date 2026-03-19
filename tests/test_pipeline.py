"""Unit tests for the traffic sign recognition pipeline.

Tests cover model architecture, data transforms, OpenCV preprocessing,
ensemble prediction logic, and configuration consistency.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import DEVICE, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES, NUM_ENSEMBLE
from src.data import GaussianNoise, OpenCVPreprocess, get_train_transforms, get_val_transforms
from src.model import create_ensemble, create_model, ensemble_predict


# ---- Model tests ----

class TestModel:
    def test_output_shape(self):
        """Model output should be (batch_size, NUM_CLASSES)."""
        model = create_model()
        model.eval()
        dummy = torch.randn(4, 3, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (4, NUM_CLASSES)

    def test_output_range_after_softmax(self):
        """Softmax output probabilities should sum to ~1 per sample."""
        model = create_model()
        model.eval()
        dummy = torch.randn(2, 3, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)
        with torch.no_grad():
            out = model(dummy)
        probs = torch.softmax(out, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_frozen_early_layers(self):
        """Early layers should be frozen (requires_grad=False)."""
        model = create_model()
        first_param = list(model.parameters())[0]
        assert first_param.requires_grad is False

    def test_final_layer_trainable(self):
        """Final classification layer should be trainable."""
        model = create_model()
        fc_params = list(model.fc.parameters())
        assert all(p.requires_grad for p in fc_params)

    def test_ensemble_size(self):
        """create_ensemble should return NUM_ENSEMBLE models."""
        ensemble = create_ensemble()
        assert len(ensemble) == NUM_ENSEMBLE

    def test_single_image_inference(self):
        """Model should handle a single image (batch_size=1)."""
        model = create_model()
        model.eval()
        dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, NUM_CLASSES)


# ---- Ensemble tests ----

class TestEnsemble:
    def test_ensemble_predict_shape(self):
        """Ensemble prediction should have shape (batch, NUM_CLASSES)."""
        models_list = create_ensemble()
        dummy = torch.randn(4, 3, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)
        with torch.no_grad():
            probs = ensemble_predict(models_list, dummy)
        assert probs.shape == (4, NUM_CLASSES)

    def test_ensemble_predict_probabilities(self):
        """Ensemble output should be valid probabilities (sum to ~1)."""
        models_list = create_ensemble()
        dummy = torch.randn(2, 3, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)
        with torch.no_grad():
            probs = ensemble_predict(models_list, dummy)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_ensemble_predict_nonnegative(self):
        """All ensemble probabilities should be non-negative."""
        models_list = create_ensemble()
        dummy = torch.randn(4, 3, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)
        with torch.no_grad():
            probs = ensemble_predict(models_list, dummy)
        assert (probs >= 0).all()


# ---- Data / preprocessing tests ----

class TestPreprocessing:
    def _make_dummy_image(self, w=64, h=64):
        """Create a random RGB PIL image."""
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_opencv_preprocess_output_type(self):
        """OpenCVPreprocess should return a PIL Image."""
        img = self._make_dummy_image()
        preprocessor = OpenCVPreprocess()
        result = preprocessor(img)
        assert isinstance(result, Image.Image)

    def test_opencv_preprocess_preserves_size(self):
        """OpenCVPreprocess should not change image dimensions."""
        img = self._make_dummy_image(100, 80)
        preprocessor = OpenCVPreprocess()
        result = preprocessor(img)
        assert result.size == img.size  # PIL size is (width, height)

    def test_opencv_preprocess_output_mode(self):
        """OpenCVPreprocess should return an RGB image."""
        img = self._make_dummy_image()
        preprocessor = OpenCVPreprocess()
        result = preprocessor(img)
        assert result.mode == "RGB"

    def test_train_transforms_output_shape(self):
        """Training transforms should produce (3, IMG_HEIGHT, IMG_WIDTH) tensor."""
        img = self._make_dummy_image(100, 100)
        transform = get_train_transforms()
        tensor = transform(img)
        assert tensor.shape == (3, IMG_HEIGHT, IMG_WIDTH)

    def test_val_transforms_output_shape(self):
        """Validation transforms should produce (3, IMG_HEIGHT, IMG_WIDTH) tensor."""
        img = self._make_dummy_image(100, 100)
        transform = get_val_transforms()
        tensor = transform(img)
        assert tensor.shape == (3, IMG_HEIGHT, IMG_WIDTH)

    def test_gaussian_noise_preserves_shape(self):
        """GaussianNoise should not change tensor shape."""
        tensor = torch.rand(3, 32, 32)
        noisy = GaussianNoise(std=0.1)(tensor)
        assert noisy.shape == tensor.shape

    def test_gaussian_noise_clamps_to_valid_range(self):
        """GaussianNoise output should be in [0, 1]."""
        tensor = torch.rand(3, 32, 32)
        noisy = GaussianNoise(std=0.5)(tensor)
        assert noisy.min() >= 0.0
        assert noisy.max() <= 1.0

    def test_val_transforms_deterministic(self):
        """Validation transforms should be deterministic (no random augmentation)."""
        img = self._make_dummy_image(100, 100)
        transform = get_val_transforms()
        t1 = transform(img)
        t2 = transform(img)
        assert torch.allclose(t1, t2, atol=1e-6)


# ---- Config tests ----

class TestConfig:
    def test_num_classes_positive(self):
        assert NUM_CLASSES > 0

    def test_image_dimensions_positive(self):
        assert IMG_HEIGHT > 0
        assert IMG_WIDTH > 0

    def test_ensemble_size_positive(self):
        assert NUM_ENSEMBLE > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
