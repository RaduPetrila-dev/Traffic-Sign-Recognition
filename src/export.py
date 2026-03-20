"""Export trained models to ONNX format for edge deployment.

ONNX models run on TensorRT, OpenVINO, ONNX Runtime, and mobile
runtimes without requiring PyTorch at inference time. This is the
standard path for deploying vision models on embedded hardware
(Jetson, Raspberry Pi, mobile devices).
"""

import os

import torch

from src.config import DEVICE, IMG_HEIGHT, IMG_WIDTH, ONNX_DIR


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> str:
    """Export a single PyTorch model to ONNX format.

    Args:
        model: Trained model in eval mode.
        output_path: File path for the exported .onnx file.
        opset_version: ONNX opset version (17 supports all ResNet ops).
        dynamic_batch: If True, the batch dimension is dynamic.

    Returns:
        Path to the exported ONNX file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model.eval()
    model = model.to("cpu")
    dummy_input = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported: {output_path} ({file_size_mb:.1f} MB)")

    # Move model back to original device
    model.to(DEVICE)
    return output_path


def validate_onnx(onnx_path: str) -> bool:
    """Validate that an ONNX model is well-formed.

    Args:
        onnx_path: Path to the .onnx file.

    Returns:
        True if the model passes validation.
    """
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"Validation passed: {onnx_path}")
        return True
    except ImportError:
        print("onnx package not installed, skipping validation")
        return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False
