#!/usr/bin/env python3
"""Dependency-light validation for the Phase 3A one-step IPC contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCHEMA_VERSION = "evla.openvla.single_step.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite IPC artifact: {path}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_request(request: dict[str, Any], request_path: Path) -> tuple[Path, np.ndarray]:
    required = {
        "schema_version", "experiment_id", "request_id", "checkpoint_revision", "unnorm_key",
        "instruction", "prompt", "image_path", "image_sha256", "image_shape", "image_dtype",
        "preprocess", "environment_steps_requested",
    }
    missing = required - request.keys()
    if missing:
        raise KeyError(f"request missing fields: {sorted(missing)}")
    if request["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unexpected schema: {request['schema_version']}")
    if request["environment_steps_requested"] != 1:
        raise ValueError("bridge request must authorize exactly one environment step")
    if request["preprocess"] != "rotate_180_then_openvla_center_crop_90_percent":
        raise ValueError("unexpected image preprocessing contract")
    image_path = (request_path.parent / request["image_path"]).resolve()
    if image_path.parent != request_path.parent.resolve():
        raise ValueError("image must remain inside the IPC directory")
    if sha256(image_path) != request["image_sha256"]:
        raise ValueError("request image hash mismatch")
    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"))
    if list(array.shape) != request["image_shape"] or request["image_shape"] != [128, 128, 3]:
        raise ValueError(f"invalid image shape: {array.shape}")
    if str(array.dtype) != request["image_dtype"] or array.dtype != np.uint8:
        raise ValueError(f"invalid image dtype: {array.dtype}")
    return image_path, array


def validate_response(response: dict[str, Any], request: dict[str, Any]) -> np.ndarray:
    required = {
        "schema_version", "classification", "experiment_id", "request_id", "request_sha256",
        "checkpoint_revision", "raw_action", "libero_adapted_action", "action_shape", "action_finite",
        "model_load_s", "inference_s", "passive_vram_allocated_bytes", "passive_vram_reserved_bytes",
        "peak_vram_allocated_bytes", "peak_vram_reserved_bytes", "worker_rss_kib_at_result",
    }
    missing = required - response.keys()
    if missing:
        raise KeyError(f"response missing fields: {sorted(missing)}")
    for key in ["schema_version", "experiment_id", "request_id", "checkpoint_revision"]:
        if response[key] != request[key]:
            raise ValueError(f"response {key} mismatch")
    if response["classification"] != "PASS":
        raise ValueError(f"worker did not PASS: {response['classification']}")
    action = np.asarray(response["libero_adapted_action"], dtype=np.float64)
    raw_action = np.asarray(response["raw_action"], dtype=np.float64)
    if action.shape != (7,) or raw_action.shape != (7,) or response["action_shape"] != [7]:
        raise ValueError("worker action shape mismatch")
    if not np.isfinite(action).all() or not np.isfinite(raw_action).all() or not response["action_finite"]:
        raise ValueError("worker produced non-finite action")
    if not np.all(action >= -1.0) or not np.all(action <= 1.0):
        raise ValueError(f"adapted action outside LIBERO range: {action}")
    if action[-1] not in (-1.0, 1.0):
        raise ValueError("adapted gripper action is not binary")
    return action.astype(np.float32)
