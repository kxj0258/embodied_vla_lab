"""OpenVLA persistent-worker adapter.

The model remains isolated in the existing ``evla-openvla`` process.  This
adapter only owns the canonical observation -> request -> validated action
boundary and never imports OpenVLA in the coordinator process.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .adapters import CanonicalObservation
from .schema import ActionSpec


class OpenVLAPolicy:
    name = "openvla_base"
    action_spec = ActionSpec(dim=7, representation="delta_pose_gripper", coordinate_frame="libero_relative", control_frequency_hz=20)

    def __init__(self, python: str, worker: Path, revision: str, unnorm_key: str, ipc_dir: Path, experiment_id: str, instruction: str, timeout_s: float = 30.0) -> None:
        self.worker = subprocess.Popen([python, str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.revision, self.unnorm_key, self.ipc_dir = revision, unnorm_key, ipc_dir
        self.experiment_id, self.instruction, self.timeout_s = experiment_id, instruction, timeout_s
        self.ipc_dir.mkdir(parents=True, exist_ok=False)
        ready = self._read_event(180.0)
        if ready.get("event") != "ready" or ready.get("classification") != "PASS":
            raise RuntimeError(f"OpenVLA worker failed to start: {ready}")
        self.ready = ready
        self.predictions = 0

    def _read_event(self, timeout_s: float) -> dict[str, Any]:
        if self.worker.stdout is None:
            raise RuntimeError("worker stdout is unavailable")
        started = time.perf_counter()
        while time.perf_counter() - started < timeout_s:
            line = self.worker.stdout.readline()
            if line:
                return json.loads(line)
            if self.worker.poll() is not None:
                raise RuntimeError(f"worker exited with {self.worker.returncode}")
        raise TimeoutError("OpenVLA worker response timeout")

    def reset(self) -> None:
        pass

    def predict(self, observation: CanonicalObservation) -> np.ndarray:
        from scripts.phase3a_openvla_bridge_protocol import load_json, sha256, validate_response, write_json_atomic

        image = np.asarray(observation.images["agentview"])
        if image.shape != (128, 128, 3) or image.dtype != np.uint8:
            raise ValueError("OpenVLA expects a uint8 128x128x3 agentview image")
        request_id = f"{self.experiment_id}:prediction:{self.predictions:06d}"
        episode_dir = self.ipc_dir / f"prediction_{self.predictions:06d}"
        episode_dir.mkdir()
        image_path = episode_dir / "agentview.png"
        Image.fromarray(image[::-1, ::-1].copy(), mode="RGB").save(image_path)
        request_path, response_path = episode_dir / "request.json", episode_dir / "response.json"
        instruction = observation.instruction or self.instruction
        request = {
            "schema_version": "evla.openvla.single_step.v1", "experiment_id": self.experiment_id,
            "request_id": request_id, "checkpoint_revision": self.revision, "unnorm_key": self.unnorm_key,
            "instruction": instruction, "prompt": f"In: What action should the robot take to {instruction.lower()}?\nOut:",
            "image_path": image_path.name, "image_sha256": sha256(image_path), "image_shape": [128, 128, 3],
            "image_dtype": "uint8", "preprocess": "rotate_180_then_openvla_center_crop_90_percent",
            "environment_steps_requested": 1,
        }
        write_json_atomic(request_path, request)
        assert self.worker.stdin is not None
        self.worker.stdin.write(json.dumps({"command": "predict", "request": str(request_path), "response": str(response_path)}) + "\n")
        self.worker.stdin.flush()
        event = self._read_event(self.timeout_s)
        if event.get("event") != "prediction" or event.get("request_id") != request_id:
            raise RuntimeError(f"unexpected OpenVLA event: {event}")
        response = load_json(response_path)
        if response.get("request_sha256") != sha256(request_path):
            raise ValueError("OpenVLA request hash mismatch")
        action = validate_response(response, request)
        self.predictions += 1
        return action

    def close(self) -> None:
        if self.worker.poll() is None and self.worker.stdin is not None:
            self.worker.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
            self.worker.stdin.flush()
            event = self._read_event(30.0)
            if event.get("event") != "shutdown" or event.get("predictions") != self.predictions:
                raise RuntimeError(f"OpenVLA shutdown mismatch: {event}")
            self.worker.wait(timeout=30)
