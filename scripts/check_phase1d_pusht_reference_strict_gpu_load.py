#!/usr/bin/env python
"""Strictly load the frozen legacy PushT Diffusion core weights and move them to CUDA."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

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
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root = args.checkpoint_dir.resolve()
    model_path = root / "model.safetensors"
    report = {
        "gate": "WP1.1d_strict_gpu_weight_load",
        "checkpoint_dir": str(root),
        "requested_device": args.device,
        "network_used": False,
        "forward_executed": False,
        "environment_constructed": False,
        "processor_constructed": False,
        "migration_executed": False,
        "optimizer_constructed": False,
    }

    if not model_path.is_file():
        report["status"] = "FAIL_MODEL_MISSING"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    if not torch.cuda.is_available():
        report.update(
            {
                "cuda_available": False,
                "status": "FAIL_CUDA_UNAVAILABLE",
            }
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 4

    device = torch.device(args.device)
    if device.type != "cuda":
        report["status"] = "FAIL_NON_CUDA_DEVICE_REQUESTED"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    torch.cuda.set_device(device)
    report.update(
        {
            "cuda_available": True,
            "cuda_device_index": device.index if device.index is not None else torch.cuda.current_device(),
            "cuda_device_name": torch.cuda.get_device_name(device),
        }
    )

    config = PreTrainedConfig.from_pretrained(root, local_files_only=True)
    config.device = "cpu"
    config.pretrained_path = None

    started = time.perf_counter()
    policy = get_policy_class(config.type)(config)
    report["policy_construction_cpu_s"] = time.perf_counter() - started

    started = time.perf_counter()
    archived_state = load_file(model_path, device="cpu")
    report["archive_read_cpu_s"] = time.perf_counter() - started
    normalization_keys = sorted(
        key for key in archived_state if key.startswith(NORMALIZATION_PREFIXES)
    )
    core_state = {
        key.replace("._orig_mod", ""): value
        for key, value in archived_state.items()
        if key not in normalization_keys
    }
    report.update(
        {
            "archived_tensor_count": len(archived_state),
            "removed_legacy_normalization_tensor_count": len(normalization_keys),
            "core_tensor_count": len(core_state),
        }
    )

    started = time.perf_counter()
    incompatible = policy.load_state_dict(core_state, strict=True)
    report.update(
        {
            "strict_load_cpu_s": time.perf_counter() - started,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "strict_load_succeeded": not incompatible.missing_keys and not incompatible.unexpected_keys,
        }
    )

    del archived_state, core_state
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    policy.to(device)
    torch.cuda.synchronize(device)
    report["cpu_to_cuda_s"] = time.perf_counter() - started

    parameter_devices = sorted({str(parameter.device) for parameter in policy.parameters()})
    buffer_devices = sorted({str(buffer.device) for buffer in policy.buffers()})
    report.update(
        {
            "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            "parameter_devices": parameter_devices,
            "buffer_devices": buffer_devices,
            "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(device),
            "cuda_memory_reserved_bytes": torch.cuda.memory_reserved(device),
            "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "maximum_resident_set_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    )

    expected_device = str(device)
    fully_on_requested_cuda = parameter_devices == [expected_device] and all(
        buffer_device == expected_device for buffer_device in buffer_devices
    )
    report["fully_on_requested_cuda"] = fully_on_requested_cuda

    if report["strict_load_succeeded"] and fully_on_requested_cuda:
        report["status"] = "PASS_STRICT_CORE_WEIGHT_LOAD_AND_CUDA_TRANSFER"
        exit_code = 0
    else:
        report["status"] = "FAIL_STRICT_LOAD_OR_CUDA_TRANSFER"
        exit_code = 3

    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
