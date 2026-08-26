#!/usr/bin/env python
"""Verify exact core weights and local loadability of the Phase 1D migrated bundle."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.migrate_policy_normalization import clean_state_dict, remove_normalization_layers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args()
    source, bundle = args.source_dir.resolve(), args.bundle_dir.resolve()

    source_state = load_file(source / "model.safetensors", device="cpu")
    source_core = clean_state_dict(remove_normalization_layers(source_state), remove_str="._orig_mod")
    bundle_state = load_file(bundle / "model.safetensors", device="cpu")
    keys_match = set(source_core) == set(bundle_state)
    unequal = sorted(
        key for key in set(source_core) & set(bundle_state) if not torch.equal(source_core[key], bundle_state[key])
    )
    source_count, bundle_count = len(source_core), len(bundle_state)
    del source_state, source_core, bundle_state
    gc.collect()

    config = PreTrainedConfig.from_pretrained(bundle, local_files_only=True)
    config.device = "cpu"
    policy = get_policy_class(config.type).from_pretrained(bundle, config=config, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(bundle), local_files_only=True
    )
    required = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
    present = {path.name for path in bundle.iterdir() if path.is_file()}
    report = {
        "gate": "WP1.1f_migrated_bundle_verification",
        "status": "PASS_MIGRATED_BUNDLE_VERIFIED",
        "network_used": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "forward_executed": False,
        "environment_constructed": False,
        "source_core_tensor_count": source_count,
        "bundle_tensor_count": bundle_count,
        "tensor_keys_match": keys_match,
        "unequal_tensor_keys": unequal,
        "required_files_missing": sorted(required - present),
        "policy_loaded": policy is not None,
        "preprocessor_loaded": preprocessor is not None,
        "postprocessor_loaded": postprocessor is not None,
    }
    passed = keys_match and not unequal and not report["required_files_missing"]
    if not passed:
        report["status"] = "FAIL_MIGRATED_BUNDLE_VERIFICATION"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
