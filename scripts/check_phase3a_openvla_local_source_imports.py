#!/usr/bin/env python3
"""Check local OpenVLA HF class imports without loading weights or initializing CUDA."""

import json
import sys
import types
from importlib.metadata import version
from pathlib import Path

import draccus
import rich
import torch


ROOT = Path(__file__).resolve().parents[1]
OPENVLA = ROOT / "openvla"


def register_inference_only_namespace() -> None:
    """Avoid importing prismatic's training/RLDS package initializers."""
    paths = {
        "prismatic": OPENVLA / "prismatic",
        "prismatic.extern": OPENVLA / "prismatic/extern",
        "prismatic.extern.hf": OPENVLA / "prismatic/extern/hf",
    }
    for name, path in paths.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module


register_inference_only_namespace()

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor  # noqa: E402


print(
    json.dumps(
        {
            "classification": "PASS",
            "draccus": version("draccus"),
            "rich": version("rich"),
            "classes": [
                OpenVLAConfig.__name__,
                OpenVLAForActionPrediction.__name__,
                PrismaticImageProcessor.__name__,
                PrismaticProcessor.__name__,
            ],
            "cuda_initialized": torch.cuda.is_initialized(),
            "model_loaded": False,
        }
    )
)
