"""Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet18.

Generates heatmaps showing which image regions the model attends to
when making a classification decision. Critical for safety-critical
systems where understanding failure modes matters as much as accuracy.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
"""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.config import GRADCAM_TARGET_LAYER, IMG_HEIGHT, IMG_WIDTH


class GradCAM:
    """Compute Grad-CAM heatmaps for a given model and target layer.

    Registers forward and backward hooks on the target layer to capture
    activations and gradients. The heatmap is computed as the ReLU of
    the weighted sum of activation channels, where weights are the
    global-average-pooled gradients.

    Args:
        model: Trained nn.Module (must be in eval mode).
        target_layer_name: Name of the layer to visualise (e.g. "layer4").
    """

    def __init__(self, model: nn.Module, target_layer_name: str = GRADCAM_TARGET_LAYER):
        self.model = model
        self.model.eval()

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks on the target layer
        target_layer = dict(model.named_modules())[target_layer_name]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input_, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate a Grad-CAM heatmap for a single image.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W).
            target_class: Class index to visualise. If None, uses the
                predicted class (the class with highest logit).

        Returns:
            Heatmap as a numpy array (H, W) with values in [0, 1].
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero all gradients, then backpropagate from the target class score
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Global average pooling of gradients -> channel weights
        gradients = self._gradients[0]  # (C, H_feat, W_feat)
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted sum of activation maps
        activations = self._activations[0]  # (C, H_feat, W_feat)
        cam = torch.zeros(activations.shape[1:], dtype=activations.dtype,
                          device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU: only keep positive contributions
        cam = torch.relu(cam)

        # Normalise to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image dimensions
        cam = cv2.resize(cam, (IMG_WIDTH, IMG_HEIGHT))
        return cam

    def overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colourmap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay a Grad-CAM heatmap on the original image.

        Args:
            image: Original RGB image as uint8 numpy array (H, W, 3).
            heatmap: Grad-CAM heatmap (H, W) with values in [0, 1].
            alpha: Blending factor (0 = only image, 1 = only heatmap).
            colourmap: OpenCV colourmap for the heatmap.

        Returns:
            Blended RGB image as uint8 numpy array (H, W, 3).
        """
        # Resize image to match heatmap if needed
        if image.shape[:2] != heatmap.shape:
            image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))

        # Convert heatmap to colour
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colour = cv2.applyColorMap(heatmap_uint8, colourmap)
        heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)

        # Blend
        blended = np.uint8(alpha * heatmap_colour + (1 - alpha) * image)
        return blended


def generate_gradcam_for_image(
    model: nn.Module,
    image_pil: Image.Image,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer: str = GRADCAM_TARGET_LAYER,
) -> tuple:
    """Convenience function: generate Grad-CAM overlay for a single image.

    Args:
        model: Trained model in eval mode.
        image_pil: Original PIL image (before preprocessing).
        input_tensor: Preprocessed tensor ready for the model.
        target_class: Class to visualise (None = predicted class).
        target_layer: Name of the target layer.

    Returns:
        tuple: (overlay_image, heatmap, predicted_class, confidence)
    """
    cam = GradCAM(model, target_layer)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = probs.max(dim=1)

    pred_class = predicted.item()
    conf = confidence.item()

    if target_class is None:
        target_class = pred_class

    # Generate heatmap
    heatmap = cam.generate(input_tensor, target_class)

    # Create overlay on original image
    original_rgb = np.array(image_pil.resize((IMG_WIDTH, IMG_HEIGHT)))
    overlay = cam.overlay(original_rgb, heatmap)

    return overlay, heatmap, pred_class, conf
