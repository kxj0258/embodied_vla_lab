#!/usr/bin/env python3
"""CUDA-disabled static validation for the persistent-worker single-episode gate."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase3a_openvla_bridge_protocol import SCHEMA_VERSION, sha256, write_json_atomic  # noqa: E402

CONFIG = ROOT / "configs/eval/openvla_libero_phase3a_single_episode.yaml"
COORDINATOR = ROOT / "scripts/evaluate_phase3a_openvla_libero_single_episode.py"
WORKER = ROOT / "scripts/run_phase3a_openvla_persistent_worker.py"


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    assert config["protocol"]["episodes"] == 1
    assert config["protocol"]["episode_horizon"] == 600
    assert config["protocol"]["init_state_index"] == 0
    assert config["protocol"]["schema_version"] == SCHEMA_VERSION
    assert config["model"]["revision"] == "fa5ae1e7509348889295bba8e08621d8b55e9baf"
    assert not (ROOT / config["output_dir"]).exists(), "formal episode output already exists"
    coordinator_source = COORDINATOR.read_text()
    worker_source = WORKER.read_text()
    ast.parse(coordinator_source, filename=str(COORDINATOR))
    ast.parse(worker_source, filename=str(WORKER))
    assert '"episodes"] != 1' in coordinator_source
    assert 'for step in range(1, int(config["protocol"]["episode_horizon"]) + 1)' in coordinator_source
    assert 'if row["terminated"] or row["truncated"]' in coordinator_source
    assert 'worker.terminate()' in coordinator_source and 'worker.kill()' in coordinator_source

    with tempfile.TemporaryDirectory(prefix="evla_phase3a_episode_static_") as directory:
        ipc = Path(directory)
        image = ipc / "image.png"
        Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB").save(image)
        request = ipc / "request.json"
        write_json_atomic(request, {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": config["experiment_id"],
            "request_id": f"{config['experiment_id']}:step:0001",
            "checkpoint_revision": config["model"]["revision"],
            "unnorm_key": "libero_goal",
            "instruction": "Turn on the stove",
            "prompt": "In: What action should the robot take to turn on the stove?\nOut:",
            "image_path": image.name,
            "image_sha256": sha256(image),
            "image_shape": [128, 128, 3],
            "image_dtype": "uint8",
            "preprocess": "rotate_180_then_openvla_center_crop_90_percent",
            "environment_steps_requested": 1,
        })
        completed = subprocess.run(
            [config["model"]["openvla_python"], str(WORKER), "--validate-request", str(request)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
            env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
        )
        assert completed.returncode == 0, completed.stderr
        response = json.loads(completed.stdout)
        assert response["classification"] == "PASS" and response["mode"] == "request_only"
    print("classification: PASS")
    print("environment_constructed: false")
    print("cuda_initialized: false")
    print("model_loaded: false")
    print("episode_executed: false")
    print("persistent-worker single-episode static contract: PASS")


if __name__ == "__main__":
    main()
