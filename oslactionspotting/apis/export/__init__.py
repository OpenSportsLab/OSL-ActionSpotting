"""Export module for OSL-ActionSpotting models.

This module provides utilities to export trained models to various formats
for production deployment, including ONNX for cross-platform inference.

Example:
    >>> from oslactionspotting.apis.export import export_to_onnx, validate_onnx_model
    >>>
    >>> # Export a trained model
    >>> export_to_onnx(model, "model.onnx", clip_len=100)
    >>>
    >>> # Validate the exported model
    >>> results = validate_onnx_model("model.onnx")
    >>> print(f"Valid: {results['valid']}")
"""

from .onnx_export import (
    export_to_onnx,
    validate_onnx_model,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)

__all__ = [
    "export_to_onnx",
    "validate_onnx_model",
    "TARGET_HEIGHT",
    "TARGET_WIDTH",
]
