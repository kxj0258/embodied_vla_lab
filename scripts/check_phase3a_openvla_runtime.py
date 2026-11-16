#!/usr/bin/env python
"""Minimal CUDA/BF16/bitsandbytes runtime gate; does not load or download OpenVLA."""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
sys.dont_write_bytecode = True

import accelerate
import bitsandbytes as bnb
import timm
import tokenizers
import torch
import transformers


def main() -> None:
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("WANDB_MODE") == "disabled"
    assert torch.cuda.is_available()
    assert torch.version.cuda == "12.8", torch.version.cuda
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    assert properties.major == 12 and properties.minor == 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    layer = bnb.nn.Linear4bit(
        16, 8, bias=False, compute_dtype=torch.bfloat16, compress_statistics=True, quant_type="nf4"
    ).to(device)
    inputs = torch.randn(2, 16, device=device, dtype=torch.bfloat16)
    outputs = layer(inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert outputs.shape == (2, 8)
    assert torch.isfinite(outputs).all()

    print(json.dumps({
        "classification": "PASS",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision_expected": "0.26.0",
        "cuda_runtime": torch.version.cuda,
        "device": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "timm": timm.__version__,
        "bitsandbytes": bnb.__version__,
        "accelerate": accelerate.__version__,
        "nf4_output_shape": list(outputs.shape),
        "nf4_output_finite": True,
        "nf4_smoke_s": elapsed,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
        "model_downloaded": False,
        "model_loaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
