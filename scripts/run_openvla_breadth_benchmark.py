#!/usr/bin/env python3
"""Run the frozen OpenVLA base over a preregistered LIBERO-Goal task grid."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evla_benchmark.evaluator import EvaluatorConfig, evaluate  # noqa: E402
from evla_benchmark.libero_adapter import LiberoGoalEnvironment  # noqa: E402
from evla_benchmark.openvla_adapter import OpenVLAPolicy  # noqa: E402
from evla_benchmark.schema import ActionSpec, ObservationSpec, TaskSpec  # noqa: E402


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs/benchmark/libero_goal_breadth_50.yaml"
    config = yaml.safe_load(config_path.read_text())
    output = ROOT / config["output_dir"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    ipc = output / "ipc"
    policy = OpenVLAPolicy(config["model"]["python"], ROOT / config["model"]["worker"], config["model"]["revision"], config["model"]["unnorm_key"], ipc, config["experiment_id"], "")
    all_episodes: list[dict] = []
    started = time.perf_counter()
    try:
        for task_cfg in config["tasks"]:
            policy.instruction = task_cfg["instruction"]
            task = TaskSpec(
                suite=config["suite"], task_id=int(task_cfg["task_id"]), name=task_cfg["name"], instruction=task_cfg["instruction"],
                observation=ObservationSpec(image_keys=("agentview",), image_shape=(128, 128, 3), state_dim=9, language_required=True),
                action=ActionSpec(dim=7, control_frequency_hz=20), episode_horizon=int(config["episode_horizon"]), split="breadth",
                init_state_indices=tuple(int(x) for x in config["states"]),
            )
            env = LiberoGoalEnvironment(config["suite"], task.task_id, 0, task.episode_horizon)
            result = evaluate(policy, env, task, EvaluatorConfig(config["experiment_id"], "openvla_base", int(config["seed"]) + task.task_id * 100, task.episode_horizon))
            for episode in result.episodes:
                all_episodes.append({**episode, "task_id": task.task_id, "task_name": task.name})
            print(json.dumps({"task_id": task.task_id, "success": result.success_count, "total": result.success_total}), flush=True)
    finally:
        policy.close()
    success_count = sum(int(x["success"]) for x in all_episodes)
    total = len(all_episodes)
    failures: dict[str, int] = {}
    for episode in all_episodes:
        category = "success" if episode["success"] else (
            "timeout" if episode.get("truncated", False) else episode.get("failure_category", "unknown")
        )
        failures[category] = failures.get(category, 0) + 1
    result = {
        "schema_version": "evla.benchmark.v1", "experiment_id": config["experiment_id"], "policy_name": "openvla_base",
        "classification": "PASS", "success_count": success_count, "success_total": total,
        "success_rate": success_count / total if total else 0.0, "episodes": all_episodes,
        "metrics": {"total_environment_steps": sum(int(x.get("steps", 0)) for x in all_episodes), "wall_s": time.perf_counter() - started},
        "failures": failures, "provenance": {"config": str(config_path), "states": config["states"], "task_count": len(config["tasks"])},
    }
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"success_rate": result["success_rate"], "failures": failures}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
