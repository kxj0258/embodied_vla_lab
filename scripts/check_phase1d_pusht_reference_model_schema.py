#!/usr/bin/env python
"""CPU-only schema comparison for the frozen legacy PushT Diffusion checkpoint."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch
from safetensors import safe_open

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class


NORMALIZATION_PREFIXES = (
    "normalize_inputs.",
    "unnormalize_outputs.",
    "normalize_targets.",
    "unnormalize_targets.",
    "normalize.",
    "unnormalize.",
    "input_normalizer.",
    "output_normalizer.",
    "normalalize_inputs.",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.checkpoint_dir.resolve()
    model_path = root / "model.safetensors"

    report = {
        "gate": "WP1.1c_strict_model_schema_compatibility",
        "checkpoint_dir": str(root),
        "device": "cpu",
        "network_used": False,
        "cuda_initialized_before": torch.cuda.is_initialized(),
        "forward_executed": False,
        "weights_loaded_into_model": False,
        "environment_constructed": False,
        "processor_constructed": False,
        "migration_executed": False,
    }

    if not model_path.is_file():
        report["status"] = "FAIL_MODEL_MISSING"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    config = PreTrainedConfig.from_pretrained(root, local_files_only=True)
    config.device = "cpu"
    config.pretrained_path = None

    started = time.perf_counter()
    policy = get_policy_class(config.type)(config)
    report["policy_construction_s"] = time.perf_counter() - started
    current_state = policy.state_dict()
    current_shapes = {key: list(value.shape) for key, value in current_state.items()}

    with safe_open(model_path, framework="pt", device="cpu") as archive:
        archived_keys = sorted(archive.keys())
        normalization_keys = sorted(
            key for key in archived_keys if key.startswith(NORMALIZATION_PREFIXES)
        )
        archived_model_shapes = {
            key.replace("._orig_mod", ""): list(archive.get_slice(key).get_shape())
            for key in archived_keys
            if key not in normalization_keys
        }

    missing_from_archive = sorted(set(current_shapes) - set(archived_model_shapes))
    unexpected_in_archive = sorted(set(archived_model_shapes) - set(current_shapes))
    shared_keys = sorted(set(current_shapes) & set(archived_model_shapes))
    shape_mismatches = [
        {
            "key": key,
            "archived": archived_model_shapes[key],
            "current": current_shapes[key],
        }
        for key in shared_keys
        if archived_model_shapes[key] != current_shapes[key]
    ]

    report.update(
        {
            "policy_type": config.type,
            "archived_tensor_count": len(archived_keys),
            "archived_normalization_tensor_count": len(normalization_keys),
            "archived_model_tensor_count": len(archived_model_shapes),
            "current_model_tensor_count": len(current_shapes),
            "shared_tensor_count": len(shared_keys),
            "missing_from_archive": missing_from_archive,
            "unexpected_in_archive": unexpected_in_archive,
            "shape_mismatches": shape_mismatches,
            "exact_key_shape_match": not (
                missing_from_archive or unexpected_in_archive or shape_mismatches
            ),
            "maximum_resident_set_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_initialized_after": torch.cuda.is_initialized(),
        }
    )

    if report["cuda_initialized_after"]:
        report["status"] = "FAIL_CUDA_SIDE_EFFECT"
        exit_code = 2
    elif report["exact_key_shape_match"]:
        report["status"] = "PASS_EXACT_MODEL_SCHEMA_MATCH"
        exit_code = 0
    else:
        report["status"] = "FAIL_MODEL_SCHEMA_MISMATCH"
        exit_code = 3

    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
