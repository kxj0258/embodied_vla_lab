#!/usr/bin/env python3
"""Run a development benchmark using the unified runtime evaluator."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evla_benchmark.act_adapter import ACTLiberoPolicy  # noqa: E402
from evla_benchmark.evaluator import EvaluatorConfig, evaluate  # noqa: E402
from evla_benchmark.libero_adapter import LiberoGoalEnvironment  # noqa: E402
from evla_benchmark.schema import ActionSpec, ObservationSpec, TaskSpec, validate_result  # noqa: E402


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs/benchmark/act_libero_dev5.yaml"
    config = yaml.safe_load(config_path.read_text())
    output = ROOT / config["output_dir"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    checkpoint = ROOT / config["checkpoint"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in payload["stats"].items()}
    task_cfg = config["task"]
    task = TaskSpec(
        suite=task_cfg["suite"], task_id=int(task_cfg["task_id"]), name=task_cfg["name"], instruction=task_cfg["instruction"],
        observation=ObservationSpec(image_keys=("agentview", "eye_in_hand"), image_shape=(128, 128, 3), state_dim=9),
        action=ActionSpec(dim=7, chunk_size=20, action_steps_per_inference=5, control_frequency_hz=20),
        episode_horizon=int(task_cfg["episode_horizon"]), split=task_cfg["split"], init_state_indices=tuple(task_cfg["init_state_indices"]),
    )
    policy = ACTLiberoPolicy(checkpoint, stats)
    environment = LiberoGoalEnvironment(task_cfg["suite"], int(task_cfg["task_id"]), int(task_cfg["task_order_index"]), task.episode_horizon)
    result = evaluate(policy, environment, task, EvaluatorConfig(config["experiment_id"], "act", int(config["protocol"]["seed"]), task.episode_horizon, float(config["protocol"]["action_low"]), float(config["protocol"]["action_high"]), bool(config["protocol"]["strict_bounds"])))
    errors = validate_result(result)
    output.mkdir(parents=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "result.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    print(json.dumps({"success_rate": result.success_rate, "failures": result.failures, "validation_errors": errors}))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
