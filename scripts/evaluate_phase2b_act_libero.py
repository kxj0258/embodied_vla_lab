#!/usr/bin/env python
"""Strict-offline single-checkpoint ACT rollout on the frozen Phase 2B LIBERO task."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/evla_phase2b_eval_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/evla_phase2b_eval_matplotlib")
sys.dont_write_bytecode = True

import numpy as np
import torch
import yaml

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.libero import LiberoEnv
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from libero.libero import benchmark


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/eval/act_libero_phase2b_ckpt100000_dev5.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_config() -> ACTConfig:
    return ACTConfig(
        input_features={
            "observation.images.agentview": PolicyFeature(FeatureType.VISUAL, (3, 128, 128)),
            "observation.images.eye_in_hand": PolicyFeature(FeatureType.VISUAL, (3, 128, 128)),
            "observation.state": PolicyFeature(FeatureType.STATE, (9,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        chunk_size=20,
        n_action_steps=5,
        use_vae=False,
        pretrained_backbone_weights=None,
        device="cuda",
    )


def policy_batch(observation: dict, stats: dict[str, np.ndarray], device: torch.device) -> dict:
    pixels = observation["pixels"]
    images = {}
    for source, target in [("agentview", "agentview"), ("eye_in_hand", "eye_in_hand")]:
        image = np.asarray(pixels[source])
        assert image.shape == (128, 128, 3) and image.dtype == np.uint8
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
        images[f"observation.images.{target}"] = torch.flip(tensor, dims=(1, 2)).unsqueeze(0).to(device)
    robot = observation["robot_state"]
    state = np.concatenate([robot["joints"]["pos"], robot["gripper"]["qpos"]]).astype(np.float32)
    assert state.shape == (9,) and np.isfinite(state).all()
    images["observation.state"] = torch.from_numpy((state - stats["sm"]) / stats["ss"]).unsqueeze(0).to(device)
    return images


def main() -> None:
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("WANDB_MODE") == "disabled"
    assert os.environ.get("LIBERO_CONFIG_PATH"), "Set the isolated project LIBERO_CONFIG_PATH"
    assert not (Path.home() / ".libero/config.yaml").exists()

    config_path = Path(os.environ.get("EVLA_EVAL_CONFIG", DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text())
    output = ROOT / config["output_dir"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    output.mkdir(parents=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    checkpoint_path = ROOT / config["checkpoint"]
    actual_hash = sha256(checkpoint_path)
    assert actual_hash == config["checkpoint_sha256"], (actual_hash, config["checkpoint_sha256"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    archived_config = checkpoint["act_config"]
    expected_config = dataclasses.asdict(policy_config())
    for key in ["chunk_size", "n_action_steps", "use_vae", "pretrained_backbone_weights"]:
        assert archived_config[key] == expected_config[key], (key, archived_config[key], expected_config[key])
    assert archived_config["input_features"] == expected_config["input_features"]
    assert archived_config["output_features"] == expected_config["output_features"]
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint["stats"].items()}

    seed = int(config["protocol"]["evaluator_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    policy = ACTPolicy(policy_config()).to(device).eval()
    policy.load_state_dict(checkpoint["model"], strict=True)

    suite = benchmark.get_benchmark_dict()[config["task"]["suite"]](
        task_order_index=config["task"]["task_order_index"]
    )
    task_id = int(config["task"]["task_id"])
    assert suite.get_task(task_id).name == config["task"]["name"]
    init_state_indices = [int(value) for value in config["protocol"]["init_state_indices"]]
    assert len(init_state_indices) == int(config["protocol"]["episodes"])
    assert init_state_indices == list(range(init_state_indices[0], init_state_indices[0] + len(init_state_indices)))
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
        episode_index=init_state_indices[0],
        n_envs=1,
        num_steps_wait=int(config["protocol"]["physics_settle_steps"]),
        control_freq=int(config["protocol"]["control_frequency_hz"]),
        control_mode="relative",
        hard_reset=True,
    )

    started = time.perf_counter()
    episodes = []
    try:
        for episode, init_index in enumerate(init_state_indices):
            observation, _ = env.reset(seed=seed + episode)
            policy.reset()
            reward_sum = 0.0
            max_reward = float("-inf")
            success = False
            inference_s = 0.0
            steps = 0
            for step in range(1, int(config["protocol"]["episode_horizon"]) + 1):
                batch = policy_batch(observation, stats, device)
                inference_started = time.perf_counter()
                normalized_action = policy.select_action(batch)
                torch.cuda.synchronize()
                inference_s += time.perf_counter() - inference_started
                action = normalized_action[0].float().cpu().numpy() * stats["as"] + stats["am"]
                action = np.clip(action, -1.0, 1.0).astype(np.float32)
                assert action.shape == (7,) and np.isfinite(action).all()
                observation, reward, terminated, truncated, info = env.step(action)
                steps = step
                reward_sum += float(reward)
                max_reward = max(max_reward, float(reward))
                success = success or bool(info.get("is_success", False))
                if terminated or truncated:
                    break
            row = {
                "episode": episode,
                "init_state_index": init_index,
                "seed": seed + episode,
                "success": success,
                "steps": steps,
                "sum_reward": reward_sum,
                "max_reward": max_reward,
                "inference_s": inference_s,
            }
            episodes.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    successes = sum(int(row["success"]) for row in episodes)
    result = {
        "classification": "PASS",
        "experiment_id": config["experiment_id"],
        "checkpoint": config["checkpoint"],
        "checkpoint_sha256": actual_hash,
        "episodes": episodes,
        "success_count": successes,
        "success_total": len(episodes),
        "success_rate": successes / len(episodes),
        "mean_sum_reward": float(np.mean([row["sum_reward"] for row in episodes])),
        "mean_max_reward": float(np.mean([row["max_reward"] for row in episodes])),
        "rollout_elapsed_s": elapsed,
        "mean_rollout_s_per_episode": elapsed / len(episodes),
        "policy_inferences": sum((row["steps"] + 4) // 5 for row in episodes),
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
        "boundary": config["boundary"],
    }
    (output / "eval_info.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
