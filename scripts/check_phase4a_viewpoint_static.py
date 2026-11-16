#!/usr/bin/env python3
"""Static, CUDA-free validation of the Phase 4A viewpoint camera gate."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/eval/phase4a_openvla_viewpoint_camera_gate_v1.yaml"
PROBE = ROOT / "scripts/probe_phase4a_libero_viewpoints.py"
RESULT = ROOT / "results/phase4a_openvla_viewpoint_camera_gate_v1_static.yaml"
EXPECTED_REVISION = "fa5ae1ed633f94762dfdd28a0a305b643b2ba8baf"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    ast.parse(PROBE.read_text(), filename=str(PROBE))
    protocol = config["protocol"]
    splits = config["splits_preregistered_for_later_rollouts"]
    checks = {
        "experiment_is_new_phase4a": config["experiment_id"].startswith("phase4a_"),
        "frozen_base_revision_exact": config["model"]["revision"] == EXPECTED_REVISION,
        "model_not_loaded_by_gate": config["model"]["weights_loaded"] is False,
        "zero_steps_and_predictions": protocol["environment_steps"] == protocol["model_predictions"] == 0,
        "single_reset": protocol["reset_count"] == 1,
        "paired_observation_keys": protocol["observation_keys"] == {
            "agentview": "agentview", "eye_in_hand": "eye_in_hand"
        },
        "paired_simulator_cameras": protocol["simulator_camera_names"] == {
            "agentview": "agentview", "eye_in_hand": "eye_in_hand"
        },
        "native_shape_frozen": protocol["observation_resolution"] == [128, 128],
        "display_shape_frozen": protocol["display_resolution"] == [640, 640],
        "common_preprocess": protocol["common_model_preprocess"] == "rotate_180_then_openvla_center_crop_90_percent",
        "task_split_disjoint": set(splits["development_task_ids"]).isdisjoint(splits["final_task_ids"]),
        "task_split_complete": sorted(splits["development_task_ids"] + splits["final_task_ids"]) == list(range(10)),
        "paired_views_exact": splits["paired_view_conditions"] == ["agentview", "eye_in_hand"],
        "episode_arithmetic": (
            len(splits["development_task_ids"]) * len(splits["init_state_indices_per_task"]) * 2
            == splits["development_episode_count"] == 30
            and len(splits["final_task_ids"]) * len(splits["init_state_indices_per_task"]) * 2
            == splits["final_episode_count"] == 30
        ),
        "fresh_output_dir": not (ROOT / config["output_dir"]).exists(),
        "fresh_structured_result": not (ROOT / config["structured_result"]).exists(),
        "fresh_raw_log": not (ROOT / config["raw_log"]).exists(),
        "probe_has_no_worker_or_cuda": all(x not in PROBE.read_text() for x in ["Popen", "CUDA_VISIBLE_DEVICES", "run_phase3a_openvla"]),
    }
    classification = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "classification": classification,
        "experiment_id": config["experiment_id"],
        "checks": checks,
        "config_sha256": sha256(CONFIG),
        "probe_sha256": sha256(PROBE),
        "cuda_calls": 0,
        "environment_resets": 0,
        "environment_steps": 0,
        "model_predictions": 0,
        "preregistered_development_episodes": 30,
        "preregistered_final_episodes": 30,
        "boundary": config["boundary"],
    }
    if RESULT.exists():
        raise FileExistsError(f"refusing overwrite: {RESULT}")
    RESULT.write_text(yaml.safe_dump(result, sort_keys=False))
    print(json.dumps(result, indent=2))
    if classification != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
