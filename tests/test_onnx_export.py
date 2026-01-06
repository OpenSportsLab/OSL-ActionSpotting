"""Unit tests for ONNX export functionality.

Run with:
    pytest tests/test_onnx_export.py
"""

import os
from typing import Any, Tuple, cast

import numpy as np
import pytest
import torch
import torch.nn as nn

from oslactionspotting.apis.export import export_to_onnx, validate_onnx_model

# --- Minimal Mock Models ---


class SimpleFrameModel(nn.Module):
    """Mimics an E2E model structure (Input: B, T, C, H, W)."""

    def __init__(self):
        super().__init__()
        # Mock the backbone structure expected by the exporter's heuristics.
        # This triggers the generation of a 5D dummy input (Video) instead of 3D.
        self.backbone = nn.Module()
        setattr(self.backbone, "_features", nn.Identity())

        # Simple 3D Convolution to ensure opset compatibility checks work
        self.conv = nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Time, Channels, Height, Width) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv(x)
        x = x.mean(dim=[-2, -1])     # Global spatial pool -> (B, 8, T)
        x = x.transpose(1, 2)        # -> (B, T, 8)
        return self.fc(x)


class SimpleFeatureModel(nn.Module):
    """Mimics a Feature-based model structure (Input: B, T, Feat)."""

    def __init__(self, input_dim: int = 512):
        super().__init__()
        # Mock attributes for heuristic checks
        self.backbone = nn.Identity()
        setattr(self.backbone, "output_dim", input_dim)
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# --- Fixtures ---


@pytest.fixture
def frame_model() -> nn.Module:
    return SimpleFrameModel().eval()


@pytest.fixture
def feature_model() -> nn.Module:
    return SimpleFeatureModel().eval()


@pytest.fixture
def onnx_path(tmp_path) -> str:
    return str(tmp_path / "model.onnx")


# --- Tests ---


@pytest.mark.parametrize("dynamic_axes", [True, False])
def test_export_basic(frame_model: nn.Module, onnx_path: str, dynamic_axes: bool):
    """Test basic export functionality with and without dynamic axes."""
    output = export_to_onnx(
        model=frame_model,
        output_path=onnx_path,
        clip_len=8,
        input_size=(32, 32),
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0


def test_export_explicit_dummy_input(frame_model: nn.Module, onnx_path: str):
    """Test exporting with a manually provided dummy input."""
    # Create a non-standard input shape to ensure manual input overrides heuristics
    custom_input = torch.randn(1, 4, 3, 16, 16)

    export_to_onnx(
        model=frame_model,
        output_path=onnx_path,
        dummy_input=custom_input,
        verbose=False
    )

    # Validate that the exported model expects the custom size (16x16)
    onnx = pytest.importorskip("onnx")
    model_proto = onnx.load(onnx_path)
    input_dim = model_proto.graph.input[0].type.tensor_type.shape.dim
    # Check width dimension (index 3 for N,T,C,H,W or N,C,T,H,W depending on layout)
    # The SimpleFrameModel takes 5 inputs.
    assert input_dim[3].dim_value == 16


@pytest.mark.parametrize("model_fixture, input_shape", [
    ("frame_model", (1, 8, 3, 32, 32)),
    ("feature_model", (1, 20, 512))
])
def test_end_to_end_accuracy(model_fixture: str, input_shape: Tuple, request, onnx_path: str):
    """
    Full integration test: Export -> Validate -> Inference -> Compare PyTorch vs ONNX.
    """
    pytest.importorskip("onnxruntime")

    model = request.getfixturevalue(model_fixture)
    input_tensor = torch.randn(*input_shape)

    # 1. Export
    export_to_onnx(
        model=model,
        output_path=onnx_path,
        dummy_input=input_tensor,
        verbose=False
    )

    # 2. Validate Structure
    res = validate_onnx_model(
        onnx_path, check_model=True, test_inference=False, verbose=False
    )
    assert res["valid"], f"Validation errors: {res['errors']}"

    # 3. Compare Inference
    import onnxruntime as ort

    # PyTorch Output
    with torch.no_grad():
        pt_out = model(input_tensor).numpy()

    # ONNX Output
    session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    # Run inference
    # Note: cast used to satisfy strict type checkers regarding ORT return types
    onnx_out_raw = session.run(None, {"input": input_tensor.numpy()})[0]
    onnx_out = cast(np.ndarray, onnx_out_raw)

    np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-4, atol=1e-5)


def test_dynamic_batch_inference(frame_model: nn.Module, onnx_path: str):
    """Ensure dynamic batch size works in the exported model."""
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    # Export with dynamic axes (default behavior)
    dummy = torch.randn(1, 5, 3, 32, 32)
    export_to_onnx(
        model=frame_model,
        output_path=onnx_path,
        dummy_input=dummy,
        verbose=False
    )

    session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    # Test with batch size 2 (different from dummy input size 1)
    batch_2_input = np.random.randn(2, 5, 3, 32, 32).astype(np.float32)

    output_raw = session.run(None, {"input": batch_2_input})[0]
    output = cast(np.ndarray, output_raw)

    assert output.shape[0] == 2


def test_error_handling(tmp_path):
    """Test that invalid inputs raise appropriate errors."""
    with pytest.raises(ValueError, match="Expected nn.Module"):
        # Explicitly cast string to Any to bypass static type checking
        # so we can test the runtime validation.
        bad_model = cast(Any, "not_a_model")
        export_to_onnx(model=bad_model, output_path=str(
            tmp_path / "fail.onnx"))

    with pytest.raises(FileNotFoundError):
        validate_onnx_model("non_existent_file.onnx")


def test_state_restoration(frame_model: nn.Module, onnx_path: str):
    """Ensure the model stays in training mode if it started there."""
    frame_model.train()
    export_to_onnx(
        model=frame_model,
        output_path=onnx_path,
        clip_len=5,
        input_size=(32, 32),
        verbose=False
    )
    assert frame_model.training is True
