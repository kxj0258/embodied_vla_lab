#!/usr/bin/env python3
"""OpenVLA-only side of the dependency-isolated Phase 3A one-step bridge."""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase3a_openvla_bridge_protocol import load_json, sha256, validate_request, write_json_atomic  # noqa: E402

OPENVLA = ROOT / "openvla"
SNAPSHOT = ROOT / "outputs/reference/phase3a_openvla_libero_goal_fa5ae1e/hf_cache/models--openvla--openvla-7b-finetuned-libero-goal/snapshots/fa5ae1e7509348889295bba8e08621d8b55e9baf"


def register_inference_only_namespace() -> None:
    for name, path in {
        "prismatic": OPENVLA / "prismatic",
        "prismatic.extern": OPENVLA / "prismatic/extern",
        "prismatic.extern.hf": OPENVLA / "prismatic/extern/hf",
    }.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--validate-request-only", action="store_true")
    args = parser.parse_args()
    request = load_json(args.request)
    image_path, _ = validate_request(request, args.request)
    if args.validate_request_only:
        print(f"PASS request-only validation: {request['request_id']}")
        return
    if args.response is None:
        raise ValueError("--response is required for inference")
    if args.response.parent.resolve() != args.request.parent.resolve():
        raise ValueError("request and response must share one IPC directory")
    if request["checkpoint_revision"] != SNAPSHOT.name:
        raise ValueError("checkpoint revision mismatch")

    import psutil
    import torch
    import transformers.modeling_utils as transformers_modeling_utils
    from accelerate.big_modeling import dispatch_model as accelerate_dispatch_model
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    register_inference_only_namespace()
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("frozen gate requires the sm_120 CUDA device")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    original_dispatch = transformers_modeling_utils.dispatch_model

    def dispatch_with_hooks(model, *dispatch_args, **dispatch_kwargs):
        dispatch_kwargs["force_hooks"] = True
        return accelerate_dispatch_model(model, *dispatch_args, **dispatch_kwargs)

    transformers_modeling_utils.dispatch_model = dispatch_with_hooks
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    ).eval()
    transformers_modeling_utils.dispatch_model = original_dispatch
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    passive_allocated = torch.cuda.memory_allocated()
    passive_reserved = torch.cuda.memory_reserved()

    image = Image.open(image_path).convert("RGB")
    crop_side = round(128 * math.sqrt(0.9))
    offset = (128 - crop_side) // 2
    image = image.crop((offset, offset, offset + crop_side, offset + crop_side)).resize(
        (128, 128), Image.Resampling.BILINEAR
    )
    inputs = processor(request["prompt"], image).to("cuda:0", dtype=torch.bfloat16)
    prompt_alignment_fix_applied = not torch.all(inputs["input_ids"][:, -1] == 29871)
    if prompt_alignment_fix_applied:
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.full((1, 1), 29871, dtype=inputs["input_ids"].dtype, device="cuda:0")],
            dim=1,
        )
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones((1, 1), dtype=inputs["attention_mask"].dtype, device="cuda:0")],
            dim=1,
        )
    if inputs["input_ids"].shape != inputs["attention_mask"].shape:
        raise AssertionError("prompt and attention mask lengths diverged")

    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        raw_action = model.predict_action(**inputs, unnorm_key=request["unnorm_key"], do_sample=False)
    torch.cuda.synchronize()
    inference_s = time.perf_counter() - inference_started
    raw_action = np.asarray(raw_action, dtype=np.float64)
    if raw_action.shape != (7,) or not np.isfinite(raw_action).all():
        raise ValueError(f"invalid raw action: {raw_action}")
    adapted = raw_action.copy()
    adapted[-1] = 2.0 * adapted[-1] - 1.0
    adapted[-1] = 1.0 if adapted[-1] > 0 else -1.0
    adapted[-1] *= -1.0
    if not np.all(adapted >= -1.0) or not np.all(adapted <= 1.0):
        raise ValueError(f"adapted action outside LIBERO range: {adapted}")

    response = {
        "schema_version": request["schema_version"],
        "classification": "PASS",
        "experiment_id": request["experiment_id"],
        "request_id": request["request_id"],
        "request_sha256": sha256(args.request),
        "checkpoint_revision": request["checkpoint_revision"],
        "raw_action": raw_action.tolist(),
        "libero_adapted_action": adapted.tolist(),
        "action_shape": [7],
        "action_finite": True,
        "prompt_alignment_fix_applied": bool(prompt_alignment_fix_applied),
        "prompt_token_count": int(inputs["input_ids"].shape[1]),
        "attention_mask_count": int(inputs["attention_mask"].shape[1]),
        "model_load_s": load_s,
        "inference_s": inference_s,
        "passive_vram_allocated_bytes": passive_allocated,
        "passive_vram_reserved_bytes": passive_reserved,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
        "worker_rss_kib_at_result": psutil.Process().memory_info().rss // 1024,
    }
    write_json_atomic(args.response, response)
    print(response)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
