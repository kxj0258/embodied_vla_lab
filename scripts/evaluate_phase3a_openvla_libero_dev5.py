#!/usr/bin/env python3
"""Bounded five-state OpenVLA development rollout with one persistent model load."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/evla_phase3a_dev5_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/evla_phase3a_dev5_matplotlib")
sys.dont_write_bytecode = True

import numpy as np
import psutil
import yaml
from PIL import Image
from lerobot.envs.libero import LiberoEnv
from libero.libero import benchmark

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_phase3a_openvla_libero_single_episode import read_worker_event, scalar  # noqa: E402
from phase3a_openvla_bridge_protocol import SCHEMA_VERSION, load_json, sha256, validate_response, write_json_atomic  # noqa: E402

CONFIG = ROOT / "configs/eval/openvla_libero_phase3a_dev5.yaml"


def main() -> None:
    if os.environ.get("LIBERO_CONFIG_PATH") != str(ROOT / "configs/libero/phase2a_runtime"):
        raise ValueError("set frozen project LIBERO_CONFIG_PATH")
    if (Path.home() / ".libero/config.yaml").exists():
        raise RuntimeError("refusing user-home LIBERO config")
    config = yaml.safe_load(Path(os.environ.get("EVLA_DEV5_CONFIG", CONFIG)).read_text())
    indices = [int(x) for x in config["protocol"]["init_state_indices"]]
    if indices != [0, 1, 2, 3, 4] or config["protocol"]["episodes"] != 5:
        raise ValueError("development gate is frozen to init states 0-4")
    if config["protocol"]["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema mismatch")
    output = ROOT / config["output_dir"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dev5 output: {output}")
    output.mkdir(parents=True)
    ipc = output / "ipc"
    ipc.mkdir()
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    seed = int(config["protocol"]["evaluator_seed"])
    random.seed(seed)
    np.random.seed(seed)
    suite = benchmark.get_benchmark_dict()[config["task"]["suite"]](task_order_index=0)
    task_id = int(config["task"]["task_id"])
    if suite.get_task(task_id).name != config["task"]["name"]:
        raise ValueError("task mismatch")
    env = LiberoEnv(
        task_suite=suite, task_id=task_id, task_suite_name=config["task"]["suite"],
        episode_length=int(config["protocol"]["episode_horizon"]),
        camera_name=["agentview_image", "robot0_eye_in_hand_image"],
        camera_name_mapping={"agentview_image": "agentview", "robot0_eye_in_hand_image": "eye_in_hand"},
        obs_type="pixels_agent_pos", observation_width=128, observation_height=128,
        init_states=True, episode_index=0, n_envs=1,
        num_steps_wait=int(config["protocol"]["physics_settle_steps"]),
        control_freq=int(config["protocol"]["control_frequency_hz"]), control_mode="relative", hard_reset=True,
    )
    worker = None
    started = time.perf_counter()
    process = psutil.Process()
    trajectory_path = output / "trajectory.jsonl"
    latest_response = None
    try:
        worker = subprocess.Popen(
            [config["model"]["openvla_python"], str(ROOT / config["model"]["worker"])],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        ready = read_worker_event(worker, int(config["protocol"]["worker_start_timeout_s"]))
        if ready.get("event") != "ready" or ready.get("classification") != "PASS":
            raise RuntimeError(f"worker readiness failed: {ready}")
        episodes = []
        all_inference = []
        total_predictions = 0
        instruction = config["task"]["instruction"]
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        with trajectory_path.open("x") as trajectory:
            for episode, init_index in enumerate(indices):
                observation, _ = env.reset(seed=seed + episode)
                reward_sum = 0.0
                max_reward = float("-inf")
                success = False
                episode_inference = 0.0
                steps = 0
                terminated_value = False
                truncated_value = False
                episode_dir = ipc / f"episode_{episode:02d}_init_{init_index:02d}"
                episode_dir.mkdir()
                episode_started = time.perf_counter()
                for step in range(1, int(config["protocol"]["episode_horizon"]) + 1):
                    image = np.asarray(observation["pixels"][config["protocol"]["camera"]])
                    if image.shape != (128, 128, 3) or image.dtype != np.uint8:
                        raise ValueError("invalid online image")
                    model_image = image[::-1, ::-1].copy()
                    image_path = episode_dir / f"step_{step:04d}_agentview_rot180.png"
                    Image.fromarray(model_image, mode="RGB").save(image_path)
                    request_path = episode_dir / f"step_{step:04d}_request.json"
                    response_path = episode_dir / f"step_{step:04d}_response.json"
                    request = {
                        "schema_version": SCHEMA_VERSION, "experiment_id": config["experiment_id"],
                        "request_id": f"{config['experiment_id']}:episode:{episode:02d}:step:{step:04d}",
                        "checkpoint_revision": config["model"]["revision"], "unnorm_key": config["model"]["unnorm_key"],
                        "instruction": instruction, "prompt": prompt, "image_path": image_path.name,
                        "image_sha256": sha256(image_path), "image_shape": [128, 128, 3], "image_dtype": "uint8",
                        "preprocess": config["protocol"]["model_image_preprocess"], "environment_steps_requested": 1,
                    }
                    write_json_atomic(request_path, request)
                    worker.stdin.write(json.dumps({"command": "predict", "request": str(request_path), "response": str(response_path)}) + "\n")
                    worker.stdin.flush()
                    event = read_worker_event(worker, int(config["protocol"]["per_action_timeout_s"]))
                    if event.get("event") != "prediction" or event.get("request_id") != request["request_id"]:
                        raise RuntimeError("worker prediction identity mismatch")
                    response = load_json(response_path)
                    latest_response = response
                    if response["request_sha256"] != sha256(request_path):
                        raise ValueError("request hash mismatch")
                    action = validate_response(response, request)
                    observation, reward, terminated, truncated, info = env.step(action)
                    steps = step
                    total_predictions += 1
                    reward_value = float(scalar(reward))
                    reward_sum += reward_value
                    max_reward = max(max_reward, reward_value)
                    success = success or bool(info.get("is_success", False))
                    terminated_value = bool(scalar(terminated))
                    truncated_value = bool(scalar(truncated))
                    inference_s = float(response["inference_s"])
                    episode_inference += inference_s
                    all_inference.append(inference_s)
                    row = {"episode": episode, "init_state_index": init_index, "step": step, "reward": reward_value,
                           "success": success, "terminated": terminated_value, "truncated": truncated_value,
                           "inference_s": inference_s, "action": response["libero_adapted_action"],
                           "image_sha256": request["image_sha256"], "response_sha256": sha256(response_path)}
                    trajectory.write(json.dumps(row) + "\n")
                    trajectory.flush()
                    print(json.dumps(row), flush=True)
                    if terminated_value or truncated_value:
                        break
                summary = {"episode": episode, "init_state_index": init_index, "seed": seed + episode,
                           "success": success, "steps": steps, "sum_reward": reward_sum, "max_reward": max_reward,
                           "terminated": terminated_value, "truncated": truncated_value,
                           "inference_s": episode_inference, "episode_wall_s": time.perf_counter() - episode_started}
                episodes.append(summary)
                print(json.dumps({"episode_summary": summary}), flush=True)

        worker.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
        worker.stdin.flush()
        shutdown = read_worker_event(worker, 30)
        if shutdown.get("event") != "shutdown" or shutdown.get("predictions") != total_predictions:
            raise RuntimeError("worker shutdown count mismatch")
        worker.wait(timeout=30)
        if worker.returncode != 0 or latest_response is None:
            raise RuntimeError(f"worker final state invalid: {worker.returncode}")
        successes = sum(int(row["success"]) for row in episodes)
        result = {
            "classification": "PASS", "experiment_id": config["experiment_id"], "execution_pass": True,
            "episodes": episodes, "success_count": successes, "success_total": 5, "success_rate": successes / 5,
            "mean_sum_reward": float(np.mean([x["sum_reward"] for x in episodes])),
            "mean_max_reward": float(np.mean([x["max_reward"] for x in episodes])),
            "total_environment_steps": sum(x["steps"] for x in episodes), "model_predictions": total_predictions,
            "model_load_s": ready["model_load_s"], "total_inference_s": float(np.sum(all_inference)),
            "mean_inference_s": float(np.mean(all_inference)), "median_inference_s": float(np.median(all_inference)),
            "p95_inference_s": float(np.percentile(all_inference, 95)), "min_inference_s": float(np.min(all_inference)),
            "max_inference_s": float(np.max(all_inference)), "rollout_end_to_end_s": time.perf_counter() - started,
            "passive_vram_allocated_bytes": ready["passive_vram_allocated_bytes"],
            "passive_vram_reserved_bytes": ready["passive_vram_reserved_bytes"],
            "peak_vram_allocated_bytes": latest_response["peak_vram_allocated_bytes"],
            "peak_vram_reserved_bytes": latest_response["peak_vram_reserved_bytes"],
            "worker_rss_kib_at_result": latest_response["worker_rss_kib_at_result"],
            "coordinator_rss_kib_at_result": process.memory_info().rss // 1024,
            "trajectory_sha256": sha256(trajectory_path), "boundary": config["boundary"],
        }
        (output / "eval_info.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try: worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill(); worker.wait(timeout=10)
        env.close()
        print(json.dumps({"cleanup": "env_closed_and_worker_reaped"}), flush=True)


if __name__ == "__main__":
    main()
