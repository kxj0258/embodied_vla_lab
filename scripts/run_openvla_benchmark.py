#!/usr/bin/env python3
"""Run the unified evaluator against the isolated OpenVLA worker."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MUJOCO_GL", "egl")

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evla_benchmark.evaluator import EvaluatorConfig, evaluate  # noqa: E402
from evla_benchmark.libero_adapter import LiberoGoalEnvironment  # noqa: E402
from evla_benchmark.openvla_adapter import OpenVLAPolicy  # noqa: E402
from evla_benchmark.schema import ActionSpec, ObservationSpec, TaskSpec, validate_result  # noqa: E402


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs/benchmark/openvla_libero_smoke.yaml"
    config = yaml.safe_load(config_path.read_text())
    output = ROOT / config["output_dir"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    task_cfg = config["task"]
    task = TaskSpec(
        suite=task_cfg["suite"], task_id=int(task_cfg["task_id"]), name=task_cfg["name"], instruction=task_cfg["instruction"],
        observation=ObservationSpec(image_keys=("agentview",), image_shape=(128, 128, 3), state_dim=9, language_required=True),
        action=ActionSpec(dim=7, control_frequency_hz=20), episode_horizon=int(task_cfg["episode_horizon"]),
        split=task_cfg["split"], init_state_indices=tuple(task_cfg["init_state_indices"]),
    )
    ipc_dir = output / "ipc"
    policy = None
    environment = None
    try:
        policy = OpenVLAPolicy(config["model"]["python"], ROOT / config["model"]["worker"], config["model"]["revision"], config["model"]["unnorm_key"], ipc_dir, config["experiment_id"], task.instruction, float(config["model"]["timeout_s"]))
        environment = LiberoGoalEnvironment(task.suite, task.task_id, int(task_cfg["task_order_index"]), task.episode_horizon)
        result = evaluate(policy, environment, task, EvaluatorConfig(config["experiment_id"], "openvla_base", int(config["protocol"]["seed"]), task.episode_horizon, float(config["protocol"]["action_low"]), float(config["protocol"]["action_high"]), bool(config["protocol"]["strict_bounds"])))
    finally:
        if policy is not None:
            policy.close()
        elif environment is not None:
            environment.close()
    errors = validate_result(result)
    # OpenVLAPolicy creates ``output/ipc`` before the first prediction so that
    # every request is written below a fresh experiment directory.  The parent
    # therefore already exists here; keep the overwrite refusal at the start
    # of main() and only make finalization idempotent.
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "result.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    print(json.dumps({"success_rate": result.success_rate, "failures": result.failures, "validation_errors": errors}))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
