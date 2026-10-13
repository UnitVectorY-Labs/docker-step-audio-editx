"""Preload PyTorch's CUDA libraries before ONNX sessions, then run the API."""
import runpy

import torch  # noqa: F401 -- also loads CUDA/cuDNN shared libraries
import onnxruntime

if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
    raise RuntimeError("GPU ONNX Runtime with CUDAExecutionProvider is required")
onnxruntime.preload_dlls()
runpy.run_path("server.py", run_name="__main__")
