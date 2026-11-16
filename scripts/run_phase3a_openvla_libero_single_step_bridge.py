#!/usr/bin/env python3
"""Run exactly one dependency-isolated OpenVLA action in one LIBERO environment."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/evla_phase3a_bridge_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/evla_phase3a_bridge_matplotlib")
sys.dont_write_bytecode = True

import numpy as np
import psutil
import yaml
from PIL import Image

from lerobot.envs.libero import LiberoEnv
from libero.libero import benchmark

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase3a_openvla_bridge_protocol import (  # noqa: E402
    SCHEMA_VERSION,
    load_json,
    sha256,
    validate_response,
    write_json_atomic,
)

DEFAULT_CONFIG = ROOT / "configs/eval/openvla_libero_phase3a_single_step_bridge.yaml"


def scalar(value):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected scalar, got shape {array.shape}")
    return array.reshape(-1)[0].item()


def main() -> None:
    if os.environ.get("LIBERO_CONFIG_PATH") != str(ROOT / "configs/libero/phase2a_runtime"):
        raise ValueError("set LIBERO_CONFIG_PATH to the frozen project runtime directory")
    if (Path.home() / ".libero/config.yaml").exists():
        raise RuntimeError("refusing to use a user-home LIBERO config")
    config_path = Path(os.environ.get("EVLA_BRIDGE_CONFIG", DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text())
    if int(config["protocol"]["environment_steps"]) != 1:
        raise ValueError("this entrypoint permits exactly one environment step")
    if config["protocol"]["schema_version"] != SCHEMA_VERSION:
        raise ValueError("config schema mismatch")
    output = ROOT / config["output_dir"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bridge output: {output}")
    output.mkdir(parents=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    seed = int(config["protocol"]["evaluator_seed"])
    random.seed(seed)
    np.random.seed(seed)
    suite = benchmark.get_benchmark_dict()[config["task"]["suite"]](
        task_order_index=int(config["task"]["task_order_index"])
    )
    task_id = int(config["task"]["task_id"])
    if suite.get_task(task_id).name != config["task"]["name"]:
        raise ValueError("task identity mismatch")
    init_state_index = int(config["protocol"]["init_state_index"])
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=config["task"]["suite"],
        episode_length=int(config["protocol"]["episode_horizon"]),
        camera_name=["agentview_image", "robot0_eye_in_hand_image"],
        camera_name_mapping={"agentview_image": "agentview", "robot0_eye_in_hand_image": "eye_in_hand"},
        obs_type="pixels_agent_pos",
        observation_width=128,
        observation_height=128,
        init_states=True,
        episode_index=init_state_index,
        n_envs=1,
        num_steps_wait=int(config["protocol"]["physics_settle_steps"]),
        control_freq=int(config["protocol"]["control_frequency_hz"]),
        control_mode="relative",
        hard_reset=True,
    )

    process = psutil.Process()
    started = time.perf_counter()
    env_step_executed = False
    try:
        observation, reset_info = env.reset(seed=seed)
        raw_image = np.asarray(observation["pixels"][config["protocol"]["camera"]])
        if raw_image.shape != (128, 128, 3) or raw_image.dtype != np.uint8:
            raise ValueError(f"unexpected online image: {raw_image.shape} {raw_image.dtype}")
        if not np.isfinite(raw_image).all():
            raise ValueError("non-finite reset image")
        # Official OpenVLA LIBERO evaluation rotates the simulator image 180 degrees.
        model_image = raw_image[::-1, ::-1].copy()
        image_path = output / "agentview_reset_rot180.png"
        Image.fromarray(model_image, mode="RGB").save(image_path)
        request_path = output / "request.json"
        response_path = output / "response.json"
        request_id = f"{config['experiment_id']}:{uuid.uuid4().hex}"
        instruction = config["task"]["instruction"]
        request = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": config["experiment_id"],
            "request_id": request_id,
            "checkpoint_revision": config["model"]["revision"],
            "unnorm_key": config["model"]["unnorm_key"],
            "instruction": instruction,
            "prompt": f"In: What action should the robot take to {instruction.lower()}?\nOut:",
            "image_path": image_path.name,
            "image_sha256": sha256(image_path),
            "image_shape": list(model_image.shape),
            "image_dtype": str(model_image.dtype),
            "preprocess": config["protocol"]["model_image_preprocess"],
            "environment_steps_requested": 1,
        }
        write_json_atomic(request_path, request)
        worker_command = [
            config["model"]["openvla_python"],
            str(ROOT / config["model"]["worker"]),
            "--request", str(request_path),
            "--response", str(response_path),
        ]
        worker_started = time.perf_counter()
        completed = subprocess.run(
            worker_command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=int(config["protocol"]["worker_timeout_s"]),
            check=False,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        worker_wall_s = time.perf_counter() - worker_started
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr, flush=True)
        if completed.returncode != 0:
            raise RuntimeError(f"OpenVLA worker exited {completed.returncode}")
        response = load_json(response_path)
        if response["request_sha256"] != sha256(request_path):
            raise ValueError("worker response references the wrong request hash")
        action = validate_response(response, request)

        step_started = time.perf_counter()
        next_observation, reward, terminated, truncated, info = env.step(action)
        env_step_s = time.perf_counter() - step_started
        env_step_executed = True
        next_image = np.asarray(next_observation["pixels"][config["protocol"]["camera"]])
        if next_image.shape != (128, 128, 3) or next_image.dtype != np.uint8:
            raise ValueError("unexpected post-step image")
        next_image_path = output / "agentview_after_step.png"
        Image.fromarray(next_image, mode="RGB").save(next_image_path)
        result = {
            "classification": "PASS",
            "experiment_id": config["experiment_id"],
            "schema_version": SCHEMA_VERSION,
            "task": config["task"],
            "seed": seed,
            "init_state_index": init_state_index,
            "environment_resets": 1,
            "environment_steps": 1,
            "reset_info": reset_info,
            "reset_observation": {
                "agentview_shape": list(raw_image.shape),
                "agentview_dtype": str(raw_image.dtype),
                "rotated_model_image_sha256": request["image_sha256"],
            },
            "raw_action": response["raw_action"],
            "libero_adapted_action": response["libero_adapted_action"],
            "action_shape": [7],
            "action_finite_and_in_bounds": True,
            "reward": float(scalar(reward)),
            "terminated": bool(scalar(terminated)),
            "truncated": bool(scalar(truncated)),
            "success": bool(info.get("is_success", False)),
            "post_step_agentview_sha256": sha256(next_image_path),
            "model_load_s": response["model_load_s"],
            "model_inference_s": response["inference_s"],
            "worker_wall_s": worker_wall_s,
            "environment_step_s": env_step_s,
            "end_to_end_s": time.perf_counter() - started,
            "passive_vram_allocated_bytes": response["passive_vram_allocated_bytes"],
            "passive_vram_reserved_bytes": response["passive_vram_reserved_bytes"],
            "peak_vram_allocated_bytes": response["peak_vram_allocated_bytes"],
            "peak_vram_reserved_bytes": response["peak_vram_reserved_bytes"],
            "worker_rss_kib_at_result": response["worker_rss_kib_at_result"],
            "coordinator_rss_kib_at_result": process.memory_info().rss // 1024,
            "request_sha256": sha256(request_path),
            "response_sha256": sha256(response_path),
            "boundary": config["boundary"],
        }
        (output / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(json.dumps(result, default=str), flush=True)
    finally:
        env.close()
        print(json.dumps({"cleanup": "env_closed", "environment_step_executed": env_step_executed}), flush=True)


if __name__ == "__main__":
    main()
