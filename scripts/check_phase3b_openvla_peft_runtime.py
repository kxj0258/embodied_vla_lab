#!/usr/bin/env python3
"""Import-only PEFT gate; no CUDA initialization or model load."""
import importlib.metadata as metadata
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phase3b_openvla_peft_runtime_gate_v1_20260827"
OUTPUT = ROOT / f"outputs/eval/{EXPERIMENT_ID}"
if OUTPUT.exists():
    raise FileExistsError(f"refusing to overwrite: {OUTPUT}")
before = torch.cuda.is_initialized()
import peft
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

assert metadata.version("peft") == "0.11.1"
assert not before and not torch.cuda.is_initialized()
assert all([LoraConfig, get_peft_model, prepare_model_for_kbit_training])
result = {
    "classification": "PASS",
    "experiment_id": EXPERIMENT_ID,
    "peft_version": metadata.version("peft"),
    "torch_version": metadata.version("torch"),
    "transformers_version": metadata.version("transformers"),
    "accelerate_version": metadata.version("accelerate"),
    "cuda_initialized": False,
    "model_loaded": False,
    "training_executed": False,
}
OUTPUT.mkdir(parents=True)
(OUTPUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
