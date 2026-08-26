#!/usr/bin/env python
"""Static, local-only audit for a PushT Diffusion reference checkpoint.

This checker never contacts the Hub, constructs a policy, initializes CUDA, or
runs an environment. It is intentionally suitable for the WP1.1 compatibility
gate before any checkpoint load or rollout is authorized.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_REFERENCE = {
    "type": "diffusion",
    "n_obs_steps": 2,
    "horizon": 16,
    "n_action_steps": 8,
    "crop_shape": [84, 84],
    "crop_is_random": True,
    "use_group_norm": True,
    "noise_scheduler_type": "DDPM",
    "num_train_timesteps": 100,
    "prediction_type": "epsilon",
}

EXPECTED_FEATURES = {
    "observation.image": {"type": "VISUAL", "shape": [3, 96, 96]},
    "observation.state": {"type": "STATE", "shape": [2]},
    "action": {"type": "ACTION", "shape": [2]},
}

NORMALIZATION_KEY_PATTERNS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Local directory containing reference checkpoint files.",
    )
    parser.add_argument(
        "--require-weights",
        action="store_true",
        help="Require model.safetensors and inspect its tensor metadata without loading tensors.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main() -> int:
    args = parse_args()
    root = args.checkpoint_dir.resolve()
    config_path = root / "config.json"
    train_config_path = root / "train_config.json"
    model_path = root / "model.safetensors"
    preprocessor_path = root / "policy_preprocessor.json"
    postprocessor_path = root / "policy_postprocessor.json"

    report: dict[str, Any] = {
        "gate": "WP1.1",
        "checkpoint_dir": str(root),
        "network_used": False,
        "cuda_initialized": False,
        "policy_constructed": False,
        "environment_constructed": False,
        "files": {},
        "contract_checks": {},
        "compatibility_risks": [],
    }

    if not root.is_dir():
        report["status"] = "FAIL_LOCAL_DIRECTORY_MISSING"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    for name, path in {
        "config": config_path,
        "train_config": train_config_path,
        "model": model_path,
        "policy_preprocessor": preprocessor_path,
        "policy_postprocessor": postprocessor_path,
    }.items():
        report["files"][name] = {"path": str(path), "present": path.is_file()}

    if not config_path.is_file():
        report["status"] = "FAIL_CONFIG_MISSING"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    config = load_json(config_path)
    for key, expected in EXPECTED_REFERENCE.items():
        actual = config.get(key)
        report["contract_checks"][key] = {
            "expected": expected,
            "actual": actual,
            "match": actual == expected,
        }

    input_features = config.get("input_features", {})
    output_features = config.get("output_features", {})
    actual_features = {
        "observation.image": input_features.get("observation.image"),
        "observation.state": input_features.get("observation.state"),
        "action": output_features.get("action"),
    }
    for key, expected in EXPECTED_FEATURES.items():
        report["contract_checks"][f"feature:{key}"] = {
            "expected": expected,
            "actual": actual_features[key],
            "match": actual_features[key] == expected,
        }

    normalization = config.get("normalization_mapping")
    report["contract_checks"]["normalization_mapping"] = {
        "expected": {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"},
        "actual": normalization,
        "match": normalization
        == {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"},
    }

    processors_present = preprocessor_path.is_file() and postprocessor_path.is_file()
    if not processors_present:
        report["compatibility_risks"].append(
            {
                "id": "legacy_processor_bundle_missing",
                "severity": "BLOCKING_FOR_DIRECT_LEROBOT_EVAL",
                "detail": (
                    "Pinned LeRobot 0.6.2 loads policy_preprocessor.json and "
                    "policy_postprocessor.json whenever --policy.path is supplied."
                ),
            }
        )

    if args.require_weights and not model_path.is_file():
        report["compatibility_risks"].append(
            {
                "id": "weights_missing",
                "severity": "BLOCKING_FOR_WEIGHT_AUDIT",
                "detail": "--require-weights was set but model.safetensors is absent.",
            }
        )

    if model_path.is_file():
        try:
            from safetensors import safe_open

            with safe_open(model_path, framework="pt", device="cpu") as archive:
                keys = set(archive.keys())
                pos_grid_key = next((key for key in keys if key.endswith("rgb_encoder.pool.pos_grid")), None)
                pos_grid_shape = list(archive.get_slice(pos_grid_key).get_shape()) if pos_grid_key else None
                report["weights"] = {
                    "tensor_count": len(keys),
                    "pos_grid_key": pos_grid_key,
                    "pos_grid_shape": pos_grid_shape,
                }
                normalization_keys = sorted(
                    key for key in keys if any(pattern in key for pattern in NORMALIZATION_KEY_PATTERNS)
                )
                report["weights"]["normalization_tensors"] = [
                    {
                        "key": key,
                        "shape": list(archive.get_slice(key).get_shape()),
                        "values": archive.get_tensor(key).tolist(),
                    }
                    for key in normalization_keys
                ]
                report["weights"]["normalization_tensor_count"] = len(normalization_keys)
                normalization_values = {
                    item["key"]: item["values"] for item in report["weights"]["normalization_tensors"]
                }
                target_min = normalization_values.get("normalize_targets.buffer_action.min")
                target_max = normalization_values.get("normalize_targets.buffer_action.max")
                output_min = normalization_values.get("unnormalize_outputs.buffer_action.min")
                output_max = normalization_values.get("unnormalize_outputs.buffer_action.max")
                report["weights"]["action_target_output_stats_match"] = {
                    "min": target_min == output_min,
                    "max": target_max == output_max,
                }
                crop_shape = config.get("crop_shape")
                image_shape = input_features.get("observation.image", {}).get("shape")
                spatial_input = crop_shape or (image_shape[1:] if image_shape else None)
                expected_pos_grid_shape = (
                    [math.ceil(spatial_input[0] / 32) * math.ceil(spatial_input[1] / 32), 2]
                    if spatial_input
                    else None
                )
                report["weights"]["expected_pos_grid_shape_for_pinned_resnet18"] = (
                    expected_pos_grid_shape
                )
                if pos_grid_shape is not None and pos_grid_shape != expected_pos_grid_shape:
                    report["compatibility_risks"].append(
                        {
                            "id": "spatial_softmax_pos_grid_mismatch",
                            "severity": "BLOCKING_PENDING_LOAD_PROOF",
                            "detail": (
                                f"Archived grid {pos_grid_shape} differs from the statically expected "
                                f"{expected_pos_grid_shape}. Do not patch or use strict=False without a "
                                "separately reviewed compatibility decision."
                            ),
                        }
                    )
        except Exception as exc:
            report["compatibility_risks"].append(
                {
                    "id": "safetensors_metadata_read_failed",
                    "severity": "BLOCKING_FOR_WEIGHT_AUDIT",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    contract_matches = all(check["match"] for check in report["contract_checks"].values())
    report["contract_matches_published_reference"] = contract_matches
    if report["compatibility_risks"]:
        report["status"] = "BLOCKED_COMPATIBILITY_REVIEW_REQUIRED"
        exit_code = 3
    elif args.require_weights and not model_path.is_file():
        report["status"] = "FAIL_WEIGHTS_MISSING"
        exit_code = 2
    elif not contract_matches:
        report["status"] = "FAIL_REFERENCE_CONTRACT_DRIFT"
        exit_code = 2
    else:
        report["status"] = "PASS_STATIC_LOCAL_AUDIT"
        exit_code = 0

    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
