#!/usr/bin/env python3
"""Load frozen OpenVLA in NF4 and predict exactly one action; no environment or training."""

import hashlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import psutil
import torch
import transformers.modeling_utils as transformers_modeling_utils
from accelerate.big_modeling import dispatch_model as accelerate_dispatch_model
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
OPENVLA = ROOT / "openvla"
SNAPSHOT = ROOT / "outputs/reference/phase3a_openvla_libero_goal_fa5ae1e/hf_cache/models--openvla--openvla-7b-finetuned-libero-goal/snapshots/fa5ae1e7509348889295bba8e08621d8b55e9baf"
FIXTURE = ROOT / "outputs/reference/phase3a_openvla_fixture_demo0_t000/agentview_rgb.png"
EXPECTED_FIXTURE_SHA256 = "d4c9852bcb00f084c104ba1381579720a154796eecbde26669d53a97a0278dbd"
EXPERIMENT_ID = "phase3a_openvla_4bit_single_image_v4_demo0_t000_seed20260827"
RESULT = ROOT / f"outputs/eval/{EXPERIMENT_ID}/result.json"

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def dispatch_quantized_model_with_hooks(model, *args, **kwargs):
    """Compatibility shim for Transformers 4.40.1 + modern Accelerate/bitsandbytes."""
    kwargs["force_hooks"] = True
    return accelerate_dispatch_model(model, *args, **kwargs)


def main() -> None:
    if RESULT.parent.exists():
        raise FileExistsError(f"refusing to overwrite output: {RESULT.parent}")
    if sha256(FIXTURE) != EXPECTED_FIXTURE_SHA256:
        raise AssertionError("fixture hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise AssertionError(f"unexpected compute capability: {torch.cuda.get_device_capability(0)}")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    process = psutil.Process()
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
    # Transformers 4.40.1 forbids `.to()` for 4-bit models, while Accelerate 1.10 assumes
    # bitsandbytes >=0.43.2 supports it. Force Accelerate hooks so dispatch never calls `.to()`.
    transformers_modeling_utils.dispatch_model = dispatch_quantized_model_with_hooks
    model = AutoModelForVision2Seq.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    passive_allocated = torch.cuda.memory_allocated()
    passive_reserved = torch.cuda.memory_reserved()

    image = Image.open(FIXTURE).convert("RGB")
    crop_side = round(128 * math.sqrt(0.9))
    offset = (128 - crop_side) // 2
    image = image.crop((offset, offset, offset + crop_side, offset + crop_side)).resize((128, 128), Image.Resampling.BILINEAR)
    prompt = "In: What action should the robot take to turn on the stove?\nOut:"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    # Upstream predict_action appends token 29871 when the prompt ends in 29901, but does not
    # extend attention_mask. Append both here so predict_action sees an already aligned prompt.
    prompt_alignment_fix_applied = not torch.all(inputs["input_ids"][:, -1] == 29871)
    if prompt_alignment_fix_applied:
        empty_token = torch.full(
            (inputs["input_ids"].shape[0], 1), 29871, dtype=inputs["input_ids"].dtype, device="cuda:0"
        )
        mask_extension = torch.ones(
            (inputs["attention_mask"].shape[0], 1),
            dtype=inputs["attention_mask"].dtype,
            device="cuda:0",
        )
        inputs["input_ids"] = torch.cat([inputs["input_ids"], empty_token], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], mask_extension], dim=1)
    if inputs["input_ids"].shape != inputs["attention_mask"].shape:
        raise AssertionError("input_ids and attention_mask must remain length-aligned")

    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        raw_action = model.predict_action(**inputs, unnorm_key="libero_goal", do_sample=False)
    torch.cuda.synchronize()
    inference_s = time.perf_counter() - inference_started
    raw_action = np.asarray(raw_action, dtype=np.float64)
    if raw_action.shape != (7,) or not np.isfinite(raw_action).all():
        raise AssertionError(f"invalid action: shape={raw_action.shape}, values={raw_action}")
    adapted = raw_action.copy()
    adapted[-1] = 2.0 * adapted[-1] - 1.0
    adapted[-1] = 1.0 if adapted[-1] > 0 else -1.0
    adapted[-1] *= -1.0

    result = {
        "classification": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "checkpoint_revision": SNAPSHOT.name,
        "fixture": str(FIXTURE.relative_to(ROOT)),
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "prompt": prompt,
        "center_crop_area": 0.9,
        "prompt_alignment_fix_applied": bool(prompt_alignment_fix_applied),
        "prompt_token_count": inputs["input_ids"].shape[1],
        "attention_mask_count": inputs["attention_mask"].shape[1],
        "quantization": "NF4_4bit",
        "compute_dtype": "bfloat16",
        "attention": "eager",
        "load_s": load_s,
        "inference_s": inference_s,
        "raw_action": raw_action.tolist(),
        "libero_adapted_action": adapted.tolist(),
        "action_finite": True,
        "passive_vram_allocated_bytes": passive_allocated,
        "passive_vram_reserved_bytes": passive_reserved,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
        "rss_kib_at_result": process.memory_info().rss // 1024,
        "boundary": "One frozen-image action prediction only; no environment step, success, reward, rollout, or training evidence.",
    }
    RESULT.parent.mkdir(parents=True)
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
