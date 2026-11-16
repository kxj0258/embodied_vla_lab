#!/usr/bin/env python
"""Source-only Phase 3A OpenVLA contract gate; no imports from the target runtime."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def require(pattern: str, text: str, source: str) -> None:
    assert re.search(pattern, text, flags=re.MULTILINE | re.DOTALL), f"missing contract in {source}: {pattern}"


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/model/openvla_libero_goal_phase3a.yaml").read_text())
    upstream = ROOT / "openvla"
    commit = subprocess.check_output(["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip()
    assert commit == cfg["upstream_code"]["commit"]
    assert cfg["repo_id"] == "openvla/openvla-7b-finetuned-libero-goal"
    assert cfg["revision"] == "fa5ae1e7509348889295bba8e08621d8b55e9baf"

    readme = (upstream / "README.md").read_text()
    evaluator = (upstream / "experiments/robot/libero/run_libero_eval.py").read_text()
    model_utils = (upstream / "experiments/robot/openvla_utils.py").read_text()
    robot_utils = (upstream / "experiments/robot/robot_utils.py").read_text()
    model = (upstream / "prismatic/extern/hf/modeling_prismatic.py").read_text()
    requirements = (ROOT / "configs/env/evla_openvla_minimal_inference.txt").read_text()

    require(r"openvla/openvla-7b-finetuned-libero-goal", readme, "README")
    require(r"cfg\.unnorm_key\s*=\s*cfg\.task_suite_name", evaluator, "LIBERO evaluator")
    require(r"center_crop:\s*bool\s*=\s*True", evaluator, "LIBERO evaluator")
    require(r"load_in_4bit=cfg\.load_in_4bit", model_utils, "model utils")
    require(r"prompt\s*=\s*f\"In: What action should the robot take to", model_utils, "model utils")
    require(r"normalize_gripper_action", robot_utils, "robot utils")
    require(r"invert_gripper_action", robot_utils, "robot utils")
    require(r"q99", model, "OpenVLA action model")
    require(r"q01", model, "OpenVLA action model")
    require(r"transformers==4\.40\.1", requirements, "curated requirements")
    require(r"bitsandbytes==0\.47\.0", requirements, "curated requirements")
    assert "tensorflow" not in requirements.lower()
    assert "flash" not in requirements.lower()
    assert "wandb" not in requirements.lower()

    print(json.dumps({
        "classification": "PASS",
        "phase": "3A",
        "upstream_commit": commit,
        "target_repo": cfg["repo_id"],
        "target_revision": cfg["revision"],
        "quantization": "NF4 4-bit",
        "action_dim": 7,
        "unnorm_key": "libero_goal",
        "center_crop": True,
        "download_executed": False,
        "gpu_initialized": False,
        "model_loaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
