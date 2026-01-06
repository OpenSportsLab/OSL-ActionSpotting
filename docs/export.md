# Exporting to ONNX: Quick Guide

This guide shows you how to convert your PyTorch model to ONNX format for faster inference.

## 1. Installation

Install the required packages:

```bash
pip install onnx onnxruntime
```

## 2. Export using Command Line (Easiest)

Run the export tool with your config file and trained weights:

```bash
python tools/export_onnx.py \
    configs/e2espot/e2espot.py \
    --weights outputs/best_model.pt \
    --output model.onnx
```

### Useful Flags
*   `--validate`: Checks if the exported file is valid.
*   `--test-inference`: Runs a dummy prediction to make sure it works.
*   `--opset 16`: Use a newer ONNX version (try this if export fails).
*   `--no-dynamic-axes`: Use this if you need a fixed batch size.

## 3. Export using Python

If you want to do it inside your code:

```python
from oslactionspotting.apis.export import export_to_onnx

# Assuming 'model' is your loaded PyTorch model
export_to_onnx(
    model=model,
    output_path="model.onnx",
    clip_len=100  # Make sure this matches your training clip length
)
```

## 4. How to Run the Model

Here is how to use the `.onnx` file to make predictions:

```python
import onnxruntime as ort
import numpy as np

# 1. Load the model
session = ort.InferenceSession("model.onnx")

# 2. Create dummy input (Batch, Time, Channels, Height, Width)
# Adjust these numbers to match your model input
input_data = np.random.randn(1, 100, 3, 224, 398).astype(np.float32)

# 3. Get input name automatically
input_name = session.get_inputs()[0].name

# 4. Run prediction
outputs = session.run(None, {input_name: input_data})

print(f"Output shape: {outputs[0].shape}")
```
