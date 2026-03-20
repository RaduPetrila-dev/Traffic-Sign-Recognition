"""Comprehensive test suite for the traffic sign recognition pipeline.

Covers: model architecture, ensemble logic, data transforms, OpenCV
preprocessing, Grad-CAM, ONNX export, labels, and configuration.

Run: python -m pytest tests/ -v
"""

import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import (
    DEVICE, GRADCAM_TARGET_LAYER, IMG_HEIGHT, IMG_WIDTH,
    NUM_CLASSES, NUM_ENSEMBLE,
)
from src.data import GaussianNoise, OpenCVPreprocess, get_train_transforms, get_val_transforms
from src.gradcam import GradCAM, generate_gradcam_for_image
from src.labels import SIGN_NAMES, get_sign_name
from src.model import create_ensemble, create_model, ensemble_predict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_image(w: int = 64, h: int = 64) -> Image.Image:
    """Create a random RGB PIL image for testing."""
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _dummy_tensor(batch: int = 2) -> torch.Tensor:
    """Create a random input tensor on the correct device."""
    return torch.randn(batch, 3, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModel:
    def test_output_shape(self):
        model = create_model()
        model.eval()
        with torch.no_grad():
            out = model(_dummy_tensor(4))
        assert out.shape == (4, NUM_CLASSES)

    def test_single_image(self):
        model = create_model()
        model.eval()
        with torch.no_grad():
            out = model(_dummy_tensor(1))
        assert out.shape == (1, NUM_CLASSES)

    def test_softmax_sums_to_one(self):
        model = create_model()
        model.eval()
        with torch.no_grad():
            out = model(_dummy_tensor(2))
        probs = torch.softmax(out, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_early_layers_frozen(self):
        model = create_model()
        first_param = list(model.parameters())[0]
        assert first_param.requires_grad is False

    def test_classification_head_trainable(self):
        model = create_model()
        fc_params = list(model.fc.parameters())
        assert all(p.requires_grad for p in fc_params)

    def test_classification_head_output_dim(self):
        model = create_model()
        last_linear = list(model.fc.children())[-1]
        assert last_linear.out_features == NUM_CLASSES

    def test_dropout_present(self):
        model = create_model()
        children = list(model.fc.children())
        dropout_layers = [c for c in children if isinstance(c, torch.nn.Dropout)]
        assert len(dropout_layers) >= 1


# ---------------------------------------------------------------------------
# Ensemble tests
# ---------------------------------------------------------------------------

class TestEnsemble:
    def test_ensemble_count(self):
        ensemble = create_ensemble()
        assert len(ensemble) == NUM_ENSEMBLE

    def test_ensemble_predict_shape(self):
        models_list = create_ensemble()
        with torch.no_grad():
            probs = ensemble_predict(models_list, _dummy_tensor(4))
        assert probs.shape == (4, NUM_CLASSES)

    def test_ensemble_probabilities_sum_to_one(self):
        models_list = create_ensemble()
        with torch.no_grad():
            probs = ensemble_predict(models_list, _dummy_tensor(3))
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_ensemble_probabilities_nonnegative(self):
        models_list = create_ensemble()
        with torch.no_grad():
            probs = ensemble_predict(models_list, _dummy_tensor(4))
        assert (probs >= 0).all()

    def test_ensemble_members_are_different(self):
        """Each ensemble member should have different random FC weights."""
        ensemble = create_ensemble()
        w0 = list(ensemble[0].fc.parameters())[-1].data
        w1 = list(ensemble[1].fc.parameters())[-1].data
        assert not torch.allclose(w0, w1)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_opencv_returns_pil(self):
        result = OpenCVPreprocess()(_dummy_image())
        assert isinstance(result, Image.Image)

    def test_opencv_preserves_size(self):
        img = _dummy_image(100, 80)
        result = OpenCVPreprocess()(img)
        assert result.size == img.size

    def test_opencv_preserves_mode(self):
        result = OpenCVPreprocess()(_dummy_image())
        assert result.mode == "RGB"

    def test_opencv_different_clip_limits(self):
        img = _dummy_image()
        r1 = np.array(OpenCVPreprocess(clip_limit=1.0)(img))
        r2 = np.array(OpenCVPreprocess(clip_limit=4.0)(img))
        assert not np.array_equal(r1, r2)

    def test_train_transforms_shape(self):
        tensor = get_train_transforms()(_dummy_image(100, 100))
        assert tensor.shape == (3, IMG_HEIGHT, IMG_WIDTH)

    def test_val_transforms_shape(self):
        tensor = get_val_transforms()(_dummy_image(100, 100))
        assert tensor.shape == (3, IMG_HEIGHT, IMG_WIDTH)

    def test_val_transforms_deterministic(self):
        img = _dummy_image(100, 100)
        t1 = get_val_transforms()(img)
        t2 = get_val_transforms()(img)
        assert torch.allclose(t1, t2, atol=1e-6)

    def test_gaussian_noise_preserves_shape(self):
        tensor = torch.rand(3, 32, 32)
        noisy = GaussianNoise(std=0.1)(tensor)
        assert noisy.shape == tensor.shape

    def test_gaussian_noise_clamps_range(self):
        tensor = torch.rand(3, 32, 32)
        noisy = GaussianNoise(std=0.5)(tensor)
        assert noisy.min() >= 0.0
        assert noisy.max() <= 1.0

    def test_gaussian_noise_zero_std_is_identity(self):
        tensor = torch.rand(3, 32, 32)
        result = GaussianNoise(mean=0.0, std=0.0)(tensor)
        assert torch.allclose(result, tensor)


# ---------------------------------------------------------------------------
# Grad-CAM tests
# ---------------------------------------------------------------------------

class TestGradCAM:
    def test_heatmap_shape(self):
        model = create_model()
        cam = GradCAM(model)
        heatmap = cam.generate(_dummy_tensor(1))
        assert heatmap.shape == (IMG_HEIGHT, IMG_WIDTH)

    def test_heatmap_range(self):
        model = create_model()
        cam = GradCAM(model)
        heatmap = cam.generate(_dummy_tensor(1))
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_heatmap_with_target_class(self):
        model = create_model()
        cam = GradCAM(model)
        heatmap = cam.generate(_dummy_tensor(1), target_class=0)
        assert heatmap.shape == (IMG_HEIGHT, IMG_WIDTH)

    def test_overlay_shape(self):
        model = create_model()
        cam = GradCAM(model)
        heatmap = cam.generate(_dummy_tensor(1))
        image = np.random.randint(0, 256, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        overlay = cam.overlay(image, heatmap)
        assert overlay.shape == (IMG_HEIGHT, IMG_WIDTH, 3)

    def test_overlay_dtype(self):
        model = create_model()
        cam = GradCAM(model)
        heatmap = cam.generate(_dummy_tensor(1))
        image = np.random.randint(0, 256, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        overlay = cam.overlay(image, heatmap)
        assert overlay.dtype == np.uint8

    def test_convenience_function(self):
        model = create_model()
        img = _dummy_image(IMG_WIDTH, IMG_HEIGHT)
        tensor = get_val_transforms()(img)
        overlay, heatmap, pred_class, conf = generate_gradcam_for_image(
            model, img, tensor
        )
        assert overlay.shape == (IMG_HEIGHT, IMG_WIDTH, 3)
        assert 0 <= pred_class < NUM_CLASSES
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# ONNX export tests
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_creates_file(self):
        from src.export import export_to_onnx
        model = create_model()
        model.eval()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            export_to_onnx(model, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_export_file_size_reasonable(self):
        from src.export import export_to_onnx
        model = create_model()
        model.eval()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            export_to_onnx(model, path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            assert 10 < size_mb < 100  # ResNet18 should be ~44 MB
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Labels tests
# ---------------------------------------------------------------------------

class TestLabels:
    def test_all_43_classes_present(self):
        assert len(SIGN_NAMES) == NUM_CLASSES

    def test_all_keys_in_range(self):
        for key in SIGN_NAMES:
            assert 0 <= key < NUM_CLASSES

    def test_get_sign_name_valid(self):
        assert get_sign_name(14) == "Stop"

    def test_get_sign_name_unknown(self):
        result = get_sign_name(999)
        assert "Unknown" in result

    def test_no_empty_names(self):
        for name in SIGN_NAMES.values():
            assert len(name.strip()) > 0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_num_classes_matches_labels(self):
        assert NUM_CLASSES == len(SIGN_NAMES)

    def test_image_dimensions_square(self):
        assert IMG_HEIGHT == IMG_WIDTH

    def test_image_dimensions_positive(self):
        assert IMG_HEIGHT > 0

    def test_ensemble_size_positive(self):
        assert NUM_ENSEMBLE > 0

    def test_gradcam_layer_exists_in_model(self):
        model = create_model()
        layer_names = [name for name, _ in model.named_modules()]
        assert GRADCAM_TARGET_LAYER in layer_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
