#!/usr/bin/env python3
"""CLI tool to export OSL-ActionSpotting models to ONNX format.

This script provides a command-line interface for exporting trained models
to ONNX format for deployment with ONNX Runtime, TensorRT, or other engines.

Example usage:
    # Export E2E model
    python tools/export_onnx.py configs/e2espot/e2espot.py \\
        --weights outputs/e2e/best_checkpoint.pt \\
        --output model.onnx

    # Export with custom settings
    python tools/export_onnx.py configs/learnablepooling/json_netvlad.py \\
        --weights outputs/netvlad/model.pth.tar \\
        --output netvlad.onnx \\
        --clip-len 50 \\
        --opset 14

    # Export and validate
    python tools/export_onnx.py configs/e2espot/e2espot.py \\
        --weights outputs/e2e/best_checkpoint.pt \\
        --output model.onnx \\
        --validate
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, Any

import torch
import torch.nn as nn
from mmengine.config import Config, DictAction
from mmengine.runner import load_checkpoint

# Local project imports
# Ensure 'oslactionspotting' is in your PYTHONPATH or installed via pip install -e .
from oslactionspotting.models.builder import build_model
from oslactionspotting.apis.export import (
    export_to_onnx,
    validate_onnx_model,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("osl_export")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export OSL-ActionSpotting models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument("config", help="Path to model config file")
    parser.add_argument("--weights", required=True,
                        help="Path to trained model weights")

    # Output path
    parser.add_argument("-o", "--output", default="model.onnx",
                        help="Output path for ONNX model")

    # Export settings
    parser.add_argument("--clip-len", type=int, default=None,
                        help="Clip length overrides config")
    parser.add_argument("--input-height", type=int, default=TARGET_HEIGHT,
                        help=f"Input H (default: {TARGET_HEIGHT})")
    parser.add_argument("--input-width", type=int, default=TARGET_WIDTH,
                        help=f"Input W (default: {TARGET_WIDTH})")
    parser.add_argument("--opset", type=int, default=14,
                        help="ONNX opset version")
    parser.add_argument("--no-dynamic-axes", action="store_true",
                        help="Fix batch size and clip length")

    # Validation
    parser.add_argument("--validate", action="store_true",
                        help="Validate structure after export")
    parser.add_argument("--test-inference", action="store_true",
                        help="Run dummy inference (requires onnxruntime)")

    # Config overrides
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config settings (e.g. --cfg-options model.backbone.type=ResNet)",
    )

    parser.add_argument("--device", default="cpu",
                        help="Device for export tracing")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    return parser.parse_args()


def get_default_args_model(cfg: Config) -> Dict[str, Any]:
    """Extract default arguments for model building from config."""
    default_args = {}
    if hasattr(cfg, "classes"):
        if isinstance(cfg.classes, list):
            default_args["classes"] = {
                c: i + 1 for i, c in enumerate(cfg.classes)}
        else:
            from oslactionspotting.core.utils.io import load_text
            classes = load_text(cfg.classes)
            default_args["classes"] = {c: i + 1 for i, c in enumerate(classes)}
    return default_args


def load_model(cfg: Config, weights_path: str, device: str = "cpu") -> nn.Module:
    """Load model architecture and weights safely.

    Args:
        cfg: Model configuration.
        weights_path: Path to model weights.
        device: Device to load model on.
    """
    logger.info(f"Building model from config: {cfg.filename}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # 1. Build Architecture
    default_args = get_default_args_model(cfg)
    model = build_model(cfg, verbose=True, default_args=default_args)

    if model is None:
        raise RuntimeError("Failed to build model (build_model returned None)")

    # 2. Load Weights (Standard MMEngine approach)
    # map_location ensures we don't try to load a GPU model onto a machine without one
    logger.info(f"Loading checkpoint: {weights_path}")
    load_checkpoint(model, weights_path, map_location="cpu", strict=True)

    # 3. Extract Implementation (E2E Wrapper Handling)
    # E2EModel usually wraps the actual implementation in `_model`
    if hasattr(model, "_model") and isinstance(model._model, nn.Module):
        logger.info("Extracting inner model (E2EModel._model) for export")
        export_model = model._model
    else:
        export_model = model

    # 4. Unwrap DataParallel
    if isinstance(export_model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        logger.info("Unwrapping DataParallel module")
        export_model = export_model.module

    # 5. Prepare for Export
    export_model.to(device)
    export_model.eval()

    return export_model


def main():
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("OSL-ActionSpotting ONNX Export Tool")
    logger.info("=" * 60)

    # Load and merge config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # Determine Clip Length logic
    # Priority: CLI Arg > Config > Default
    clip_len = args.clip_len
    if clip_len is None:
        clip_len = getattr(cfg.dataset, "clip_len", 100)
        logger.info(f"Using clip_len from config: {clip_len}")
    else:
        logger.info(f"Using clip_len from args: {clip_len}")

    try:
        # Load Model
        model = load_model(cfg, args.weights, device=args.device)

        # Export
        logger.info("-" * 60)
        output_path = export_to_onnx(
            model=model,
            output_path=args.output,
            clip_len=clip_len,
            input_size=(args.input_height, args.input_width),
            opset_version=args.opset,
            dynamic_axes=not args.no_dynamic_axes,
            device=args.device,
            verbose=args.verbose,
        )

        # Validate
        if args.validate or args.test_inference:
            logger.info("-" * 60)
            results = validate_onnx_model(
                onnx_path=output_path,
                check_model=True,
                test_inference=args.test_inference,
                clip_len=clip_len,
                input_size=(args.input_height, args.input_width),
                verbose=args.verbose,
            )

            if not results["valid"]:
                logger.error("Validation FAILED")
                for err in results["errors"]:
                    logger.error(f"  - {err}")
                sys.exit(1)
            else:
                logger.info("Validation PASSED")

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(
            f"Done in {elapsed:.2f}s. Model: {os.path.abspath(output_path)}")

    except Exception as e:
        logger.exception(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
