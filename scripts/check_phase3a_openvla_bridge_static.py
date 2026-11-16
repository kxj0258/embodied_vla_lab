#!/usr/bin/env python3
"""Static/no-CUDA validation of the Phase 3A bridge schema and worker boundary."""
from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase3a_openvla_bridge_protocol import (  # noqa: E402
    SCHEMA_VERSION,
    load_json,
    sha256,
    validate_request,
    validate_response,
    write_json_atomic,
)

CONFIG = ROOT / "configs/eval/openvla_libero_phase3a_single_step_bridge.yaml"
COORDINATOR = ROOT / "scripts/run_phase3a_openvla_libero_single_step_bridge.py"
WORKER = ROOT / "scripts/run_phase3a_openvla_bridge_worker.py"


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    assert config["protocol"]["schema_version"] == SCHEMA_VERSION
    assert config["protocol"]["environment_steps"] == 1
    assert config["protocol"]["init_state_index"] == 0
    assert config["model"]["revision"] == "fa5ae1e7509348889295bba8e08621d8b55e9baf"
    formal_output = ROOT / config["output_dir"]
    if formal_output.exists():
        formal_result = load_json(formal_output / "result.json")
        assert formal_result["experiment_id"] == config["experiment_id"]
        assert formal_result["classification"] == "PASS"
    ast.parse(COORDINATOR.read_text(), filename=str(COORDINATOR))
    ast.parse(WORKER.read_text(), filename=str(WORKER))

    with tempfile.TemporaryDirectory(prefix="evla_phase3a_bridge_static_") as directory:
        ipc = Path(directory)
        image_path = ipc / "image.png"
        Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB").save(image_path)
        request_path = ipc / "request.json"
        request = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": config["experiment_id"],
            "request_id": "static-contract-request",
            "checkpoint_revision": config["model"]["revision"],
            "unnorm_key": "libero_goal",
            "instruction": "Turn on the stove",
            "prompt": "In: What action should the robot take to turn on the stove?\nOut:",
            "image_path": image_path.name,
            "image_sha256": sha256(image_path),
            "image_shape": [128, 128, 3],
            "image_dtype": "uint8",
            "preprocess": "rotate_180_then_openvla_center_crop_90_percent",
            "environment_steps_requested": 1,
        }
        write_json_atomic(request_path, request)
        validate_request(load_json(request_path), request_path)
        completed = subprocess.run(
            [config["model"]["openvla_python"], str(WORKER), "--request", str(request_path), "--validate-request-only"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
            env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
        )
        assert completed.returncode == 0, completed.stderr
        response = {
            "schema_version": SCHEMA_VERSION,
            "classification": "PASS",
            "experiment_id": request["experiment_id"],
            "request_id": request["request_id"],
            "request_sha256": sha256(request_path),
            "checkpoint_revision": request["checkpoint_revision"],
            "raw_action": [0.0] * 7,
            "libero_adapted_action": [0.0] * 6 + [1.0],
            "action_shape": [7],
            "action_finite": True,
            "model_load_s": 0.0,
            "inference_s": 0.0,
            "passive_vram_allocated_bytes": 0,
            "passive_vram_reserved_bytes": 0,
            "peak_vram_allocated_bytes": 0,
            "peak_vram_reserved_bytes": 0,
            "worker_rss_kib_at_result": 0,
        }
        validate_response(response, request)
    print("classification: PASS")
    print("environment_constructed: false")
    print("cuda_initialized: false")
    print("model_loaded: false")
    print("environment_steps: 0")
    print("bridge static/schema/isolated-worker request validation: PASS")


if __name__ == "__main__":
    main()
