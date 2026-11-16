#!/usr/bin/env python3
"""Evaluate exactly one LIBERO episode through a persistent isolated OpenVLA worker."""
from __future__ import annotations

import json
import os
import random
import select
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/evla_phase3a_episode_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/evla_phase3a_episode_matplotlib")
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

DEFAULT_CONFIG = ROOT / "configs/eval/openvla_libero_phase3a_single_episode.yaml"


def scalar(value):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected scalar, got {array.shape}")
    return array.reshape(-1)[0].item()


def read_worker_event(worker: subprocess.Popen, timeout_s: int) -> dict:
    ready, _, _ = select.select([worker.stdout], [], [], timeout_s)
    if not ready:
        raise TimeoutError(f"OpenVLA worker response exceeded {timeout_s}s")
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError(f"OpenVLA worker closed stdout; returncode={worker.poll()}")
    event = json.loads(line)
    print(json.dumps({"worker_event": event}), flush=True)
    return event


def main() -> None:
    expected_libero_config = str(ROOT / "configs/libero/phase2a_runtime")
    if os.environ.get("LIBERO_CONFIG_PATH") != expected_libero_config:
        raise ValueError("set LIBERO_CONFIG_PATH to the frozen project runtime")
    if (Path.home() / ".libero/config.yaml").exists():
        raise RuntimeError("refusing user-home LIBERO config")
    config_path = Path(os.environ.get("EVLA_EPISODE_CONFIG", DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text())
    if config["protocol"]["episodes"] != 1 or config["protocol"]["schema_version"] != SCHEMA_VERSION:
        raise ValueError("entrypoint is frozen to exactly one episode")
    output = ROOT / config["output_dir"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite episode output: {output}")
    output.mkdir(parents=True)
    ipc = output / "ipc"
    ipc.mkdir()
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
    worker = None
    process = psutil.Process()
    started = time.perf_counter()
    trajectory_path = output / "trajectory.jsonl"
    try:
        observation, reset_info = env.reset(seed=seed)
        worker = subprocess.Popen(
            [config["model"]["openvla_python"], str(ROOT / config["model"]["worker"])],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        ready = read_worker_event(worker, int(config["protocol"]["worker_start_timeout_s"]))
        if ready.get("event") != "ready" or ready.get("classification") != "PASS":
            raise RuntimeError(f"worker readiness failed: {ready}")

        reward_sum = 0.0
        max_reward = float("-inf")
        success = False
        inference_times = []
        env_step_times = []
        steps = 0
        last_row = None
        latest_response = None
        instruction = config["task"]["instruction"]
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        with trajectory_path.open("x") as trajectory:
            for step in range(1, int(config["protocol"]["episode_horizon"]) + 1):
                image = np.asarray(observation["pixels"][config["protocol"]["camera"]])
                if image.shape != (128, 128, 3) or image.dtype != np.uint8:
                    raise ValueError(f"invalid online image at step {step}")
                model_image = image[::-1, ::-1].copy()
                image_path = ipc / f"step_{step:04d}_agentview_rot180.png"
                Image.fromarray(model_image, mode="RGB").save(image_path)
                request_path = ipc / f"step_{step:04d}_request.json"
                response_path = ipc / f"step_{step:04d}_response.json"
                request = {
                    "schema_version": SCHEMA_VERSION,
                    "experiment_id": config["experiment_id"],
                    "request_id": f"{config['experiment_id']}:step:{step:04d}",
                    "checkpoint_revision": config["model"]["revision"],
                    "unnorm_key": config["model"]["unnorm_key"],
                    "instruction": instruction,
                    "prompt": prompt,
                    "image_path": image_path.name,
                    "image_sha256": sha256(image_path),
                    "image_shape": list(model_image.shape),
                    "image_dtype": str(model_image.dtype),
                    "preprocess": config["protocol"]["model_image_preprocess"],
                    "environment_steps_requested": 1,
                }
                write_json_atomic(request_path, request)
                worker.stdin.write(json.dumps({
                    "command": "predict", "request": str(request_path), "response": str(response_path)
                }) + "\n")
                worker.stdin.flush()
                event = read_worker_event(worker, int(config["protocol"]["per_action_timeout_s"]))
                if event.get("event") != "prediction" or event.get("request_id") != request["request_id"]:
                    raise RuntimeError(f"worker prediction identity mismatch: {event}")
                response = load_json(response_path)
                latest_response = response
                if response["request_sha256"] != sha256(request_path):
                    raise ValueError("response request hash mismatch")
                action = validate_response(response, request)
                env_step_started = time.perf_counter()
                observation, reward, terminated, truncated, info = env.step(action)
                env_step_s = time.perf_counter() - env_step_started
                steps = step
                reward_value = float(scalar(reward))
                reward_sum += reward_value
                max_reward = max(max_reward, reward_value)
                success = success or bool(info.get("is_success", False))
                inference_times.append(float(response["inference_s"]))
                env_step_times.append(env_step_s)
                row = {
                    "step": step,
                    "image_sha256": request["image_sha256"],
                    "request_sha256": sha256(request_path),
                    "response_sha256": sha256(response_path),
                    "raw_action": response["raw_action"],
                    "libero_adapted_action": response["libero_adapted_action"],
                    "reward": reward_value,
                    "success": success,
                    "terminated": bool(scalar(terminated)),
                    "truncated": bool(scalar(truncated)),
                    "inference_s": response["inference_s"],
                    "environment_step_s": env_step_s,
                }
                last_row = row
                trajectory.write(json.dumps(row) + "\n")
                trajectory.flush()
                print(json.dumps(row), flush=True)
                if row["terminated"] or row["truncated"]:
                    break

        worker.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
        worker.stdin.flush()
        shutdown = read_worker_event(worker, 30)
        if shutdown.get("event") != "shutdown" or shutdown.get("predictions") != steps:
            raise RuntimeError(f"worker shutdown mismatch: {shutdown}")
        worker.wait(timeout=30)
        if worker.returncode != 0:
            raise RuntimeError(f"worker exited {worker.returncode}")
        if last_row is None or latest_response is None:
            raise RuntimeError("episode produced no environment steps")
        final_image = np.asarray(observation["pixels"][config["protocol"]["camera"]])
        final_image_path = output / "agentview_final.png"
        Image.fromarray(final_image, mode="RGB").save(final_image_path)
        result = {
            "classification": "PASS",
            "experiment_id": config["experiment_id"],
            "execution_pass": True,
            "policy_success": success,
            "task": config["task"],
            "seed": seed,
            "init_state_index": init_state_index,
            "episodes": 1,
            "steps": steps,
            "sum_reward": reward_sum,
            "max_reward": max_reward,
            "terminated": last_row["terminated"],
            "truncated": last_row["truncated"],
            "model_predictions": steps,
            "model_load_s": ready["model_load_s"],
            "mean_inference_s": float(np.mean(inference_times)),
            "median_inference_s": float(np.median(inference_times)),
            "min_inference_s": float(np.min(inference_times)),
            "max_inference_s": float(np.max(inference_times)),
            "total_inference_s": float(np.sum(inference_times)),
            "mean_environment_step_s": float(np.mean(env_step_times)),
            "episode_end_to_end_s": time.perf_counter() - started,
            "passive_vram_allocated_bytes": ready["passive_vram_allocated_bytes"],
            "passive_vram_reserved_bytes": ready["passive_vram_reserved_bytes"],
            "peak_vram_allocated_bytes": latest_response["peak_vram_allocated_bytes"],
            "peak_vram_reserved_bytes": latest_response["peak_vram_reserved_bytes"],
            "worker_rss_kib_at_result": latest_response["worker_rss_kib_at_result"],
            "coordinator_rss_kib_at_result": process.memory_info().rss // 1024,
            "trajectory_sha256": sha256(trajectory_path),
            "final_image_sha256": sha256(final_image_path),
            "reset_info": reset_info,
            "boundary": config["boundary"],
        }
        (output / "eval_info.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(json.dumps(result, default=str), flush=True)
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=10)
        env.close()
        print(json.dumps({"cleanup": "env_closed_and_worker_reaped"}), flush=True)


if __name__ == "__main__":
    main()
