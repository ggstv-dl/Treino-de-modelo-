"""Abre X-AnyLabeling usando o runtime CUDA isolado, sem alterar sua .venv."""
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
runtime = root / "runs/runtime_sam2_cuda13"
if not (runtime / "onnxruntime").is_dir():
    raise SystemExit("Runtime ausente. Consulte docs/teste_sam2_5imagens.md")
sys.path.insert(0, str(runtime))
os.environ["X_ANYLABELING_DEVICE"] = "GPU"

import torch
import onnxruntime as ort

if not torch.cuda.is_available():
    raise SystemExit("CUDA indisponível. Execute este comando no terminal do computador.")
ort.preload_dlls()
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("ONNX Runtime sem CUDAExecutionProvider")

from anylabeling.app import main

if __name__ == "__main__":
    main()
