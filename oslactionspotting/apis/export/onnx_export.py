"""ONNX export utilities for OSL-ActionSpotting models.

This module provides functions to export PyTorch models to ONNX format
for deployment with ONNX Runtime, TensorRT, or other inference engines.

Copyright 2024 OpenSportsLab Contributors
Licensed under GPL-3.0 License
"""

import logging
import os
from typing import Optional, Tuple, Union, Dict, Any, List, cast

import torch
import torch.nn as nn

# Target dimensions used in the library (from datasets/frame.py)
TARGET_HEIGHT = 224
TARGET_WIDTH = 398

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    dummy_input: Optional[torch.Tensor] = None,
    clip_len: int = 100,
    input_size: Tuple[int, int] = (TARGET_HEIGHT, TARGET_WIDTH),
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    opset_version: int = 14,
    dynamic_axes: Union[bool, Dict[str, Dict[int, str]]] = True,
    device: str = "cpu",
    verbose: bool = True,
) -> str:
    """Export an OSL-ActionSpotting model to ONNX format.

    Args:
        model (nn.Module): The PyTorch model to export.
        output_path (str): Path where the ONNX model will be saved.
        dummy_input (torch.Tensor, optional): Explicit dummy input tensor.
            If None, one will be generated based on heuristics.
        clip_len (int): Number of frames in the temporal window. Default: 100.
        input_size (Tuple[int, int]): Input frame size (H, W). Default: (224, 398).
        input_names (List[str], optional): Names for input nodes.  Default: ["input"].
        output_names (List[str], optional): Names for output nodes. Default: ["output"].
        opset_version (int): ONNX opset version. Default: 14.
        dynamic_axes (Union[bool, Dict]): If True, uses default dynamic axes configuration.
            If a Dict is provided, it is used directly. Default: True.
        device (str): Device to use for export tracing. Default: "cpu".
        verbose (bool): If True, log export progress. Default: True.

    Returns:
        str: Path to the exported ONNX model.
    """
    if not isinstance(model, nn.Module):
        raise ValueError(f"Expected nn.Module, got {type(model).__name__}")

    # Initialize mutable defaults
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Capture original state to restore later
    original_training_mode = model.training

    try:
        if verbose:
            logger.info("Exporting model to ONNX format")
            logger.info(f"  Output path: {output_path}")
            logger.info(f"  Device: {device}")

        # Prepare model
        model.to(device)
        model.eval()

        # Create dummy input if not provided
        if dummy_input is None:
            if verbose:
                logger.info(
                    f"  Generating dummy input (clip_len={clip_len}, size={input_size})")
            dummy_input = _create_dummy_input(
                model, clip_len, input_size, device)
        else:
            dummy_input = dummy_input.to(device)

        if verbose:
            logger.info(f"  Input shape: {tuple(dummy_input.shape)}")
            _log_model_info(model)

        # Configure dynamic axes
        dynamic_axes_config: Optional[Dict[str, Dict[int, str]]] = None

        if isinstance(dynamic_axes, dict):
            dynamic_axes_config = dynamic_axes
        elif dynamic_axes is True:
            # Generate default dynamic axes based on input shape
            dynamic_axes_config = _get_default_dynamic_axes(
                dummy_input, input_names, output_names
            )

        if verbose and dynamic_axes_config:
            logger.info(f"  Dynamic axes: {dynamic_axes_config}")

        # Perform ONNX export
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_input,),
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes_config,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
                verbose=False,
            )

        if verbose:
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(
                f"  Export successful! File size: {file_size_mb:.2f} MB")

    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {str(e)}") from e
    finally:
        # Restore original training mode
        model.train(original_training_mode)

    return output_path


def validate_onnx_model(
    onnx_path: str,
    check_model: bool = True,
    test_inference: bool = True,
    clip_len: int = 100,
    input_size: Tuple[int, int] = (TARGET_HEIGHT, TARGET_WIDTH),
    verbose: bool = True,
) -> Dict[str, Any]:
    """Validate an exported ONNX model.

    This function performs validation checks on an exported ONNX model,
    including structural validation and optional inference testing.

    Args:
        onnx_path (str): Path to the ONNX model file.
        check_model (bool): If True, run onnx.checker.check_model(). Default: True.
        test_inference (bool): If True, run a test inference with ONNX Runtime.
            Default: True.
        clip_len (int): Clip length for test inference. Default: 100.
        input_size (Tuple[int, int]): Input size for test inference. Default: (224, 398).
        verbose (bool): If True, log validation progress. Default: True.

    Returns:
        Dict[str, Any]: Validation results containing:
            - "valid" (bool): Overall validation status
            - "check_model" (bool): Result of onnx.checker.check_model()
            - "inference_test" (bool): Result of inference test
            - "output_shape" (tuple): Shape of inference output
            - "errors" (list): List of any error messages

    Raises:
        FileNotFoundError: If the ONNX file does not exist.
        ImportError: If required packages (onnx, onnxruntime) are not installed.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    results: Dict[str, Any] = {
        "valid": True,
        "check_model": False,
        "inference_test": False,
        "output_shape": None,
        "errors": [],
    }

    # 1. Structural Validation
    if check_model:
        try:
            import onnx
            if verbose:
                logger.info(f"Validating ONNX structure: {onnx_path}")

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            results["check_model"] = True

            if verbose:
                logger.info("  Structure validation: PASSED")
        except ImportError:
            _handle_import_error("onnx", results, verbose)
        except Exception as e:
            _handle_validation_error(
                f"Structure validation failed: {e}", results, verbose)

    # 2. Inference Validation
    if test_inference and results["valid"]:
        try:
            import onnxruntime as ort
            import numpy as np

            if verbose:
                logger.info("Testing inference with ONNX Runtime")

            session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"])

            # Introspect inputs
            input_info = session.get_inputs()[0]
            input_name = input_info.name
            input_shape = input_info.shape

            if verbose:
                logger.info(f"  Input: {input_name}, Shape: {input_shape}")

            # Generate test data
            test_shape = _resolve_dynamic_shape(
                input_shape, clip_len, input_size)
            test_input = np.random.randn(*test_shape).astype(np.float32)

            # Run inference
            outputs = session.run(None, {input_name: test_input})

            if not outputs:
                raise RuntimeError("ONNX Runtime returned no outputs")

            first_output = cast(Any, outputs[0])

            if hasattr(first_output, 'shape'):
                results["output_shape"] = tuple(first_output.shape)
                results["inference_test"] = True

                if verbose:
                    logger.info("  Inference test: PASSED")
                    logger.info(f"  Output shape: {first_output.shape}")
            else:
                logger.warning(
                    "  Inference ran, but output does not have .shape attribute")
                results["inference_test"] = True

        except ImportError:
            _handle_import_error("onnxruntime", results, verbose)
        except Exception as e:
            _handle_validation_error(
                f"Inference test failed: {e}", results, verbose)

    return results


# --- Internal Helpers ---

def _handle_import_error(pkg_name: str, results: Dict, verbose: bool) -> None:
    msg = f"{pkg_name} not installed. Run: pip install {pkg_name}"
    results["errors"].append(msg)
    results["valid"] = False
    if verbose:
        logger.warning(f"  {msg}")


def _handle_validation_error(msg: str, results: Dict, verbose: bool):
    results["errors"].append(msg)
    results["valid"] = False
    if verbose:
        logger.error(f"  {msg}")


def _create_dummy_input(
    model: nn.Module,
    clip_len: int,
    input_size: Tuple[int, int],
    device: str,
) -> torch.Tensor:
    """Heuristic to create dummy input. Prefer passing explicit dummy_input to export_to_onnx."""
    # E2E models usually have a backbone with _features
    is_e2e = False
    if hasattr(model, "backbone"):
        backbone = getattr(model, "backbone")
        if hasattr(backbone, "_features"):
            is_e2e = True

    if is_e2e:
        # Shape: (batch, clip_len, channels, height, width)
        return torch.randn(
            1, clip_len, 3, input_size[0], input_size[1],
            device=device, dtype=torch.float32
        )

    # Fallback to Feature-based model heuristic
    # Shape: (batch, window_size, feature_dim)
    feature_dim = 512
    if hasattr(model, "backbone"):
        bb = getattr(model, "backbone")
        feature_dim = getattr(
            bb, "output_dim", getattr(bb, "feature_dim", 512))
        # Handle case where dim might be a non-int (e.g. 1-element tensor)
        if hasattr(feature_dim, "item"):
            feature_dim = int(feature_dim.item())  # type: ignore

    return torch.randn(
        1, clip_len, int(feature_dim),
        device=device, dtype=torch.float32
    )


def _get_default_dynamic_axes(
    dummy_input: torch.Tensor,
    input_names: List[str],
    output_names: List[str]
) -> Dict[str, Dict[int, str]]:
    """Generate default dynamic axes based on input dimensions."""
    ndim = dummy_input.ndim

    # Define mapping strategy
    input_dyn = {}
    output_dyn = {0: "batch_size"}

    if ndim == 5:
        # (batch, time, channel, height, width)
        input_dyn = {0: "batch_size", 1: "clip_length"}
        # Assuming output mimics input temporal dim
        output_dyn = {0: "batch_size", 1: "clip_length"}
    elif ndim == 3:
        # (batch, time, feat)
        input_dyn = {0: "batch_size", 1: "window_size"}
    else:
        # Generic fallback
        input_dyn = {0: "batch_size"}

    config = {}
    for name in input_names:
        config[name] = input_dyn
    for name in output_names:
        config[name] = output_dyn

    return config


def _resolve_dynamic_shape(
    shape: List[Any],
    clip_len: int,
    input_size: Tuple[int, int],
) -> Tuple[int, ...]:
    """Resolve dynamic dimensions (strings/None) in ONNX input shape to integers."""
    resolved = []
    for i, dim in enumerate(shape):
        if isinstance(dim, (str, type(None))):
            # Resolve common dynamic axes by index
            if i == 0:
                val = 1
            elif i == 1:
                val = clip_len
            elif i == 2:
                val = 3 if len(shape) == 5 else 512  # Guessing
            elif i == 3:
                val = input_size[0]
            elif i == 4:
                val = input_size[1]
            else:
                val = 1
            resolved.append(val)
        else:
            resolved.append(int(dim))
    return tuple(resolved)


def _log_model_info(model: nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
