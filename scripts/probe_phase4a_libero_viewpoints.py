#!/usr/bin/env python3
"""One-reset, zero-step LIBERO viewpoint availability and image-integrity probe."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/evla_phase4a_viewpoint_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/evla_phase4a_viewpoint_matplotlib")
sys.dont_write_bytecode = True

import numpy as np
import yaml
from PIL import Image
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/eval/phase4a_openvla_viewpoint_camera_gate_v1.yaml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image_stats(array: np.ndarray) -> dict:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    if os.environ.get("LIBERO_CONFIG_PATH") != str(ROOT / "configs/libero/phase2a_runtime"):
        raise ValueError("set frozen project LIBERO_CONFIG_PATH")
    if (Path.home() / ".libero/config.yaml").exists():
        raise FileExistsError("refusing user-global LIBERO config")
    output = ROOT / config["output_dir"]
    result_path = ROOT / config["structured_result"]
    for target in [output, result_path]:
        if target.exists():
            raise FileExistsError(f"refusing overwrite: {target}")
    output.mkdir(parents=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    protocol = config["protocol"]
    suite = benchmark.get_benchmark_dict()[config["task"]["suite"]](task_order_index=0)
    task_id = int(config["task"]["task_id"])
    task = suite.get_task(task_id)
    if task.name != config["task"]["name"] or task.language != config["task"]["instruction"]:
        raise ValueError("task metadata mismatch")
    env = LiberoEnv(
        task_suite=suite, task_id=task_id, task_suite_name=config["task"]["suite"],
        episode_length=int(protocol["episode_horizon"]),
        camera_name=["agentview_image", "robot0_eye_in_hand_image"],
        camera_name_mapping={"agentview_image": "agentview", "robot0_eye_in_hand_image": "eye_in_hand"},
        obs_type="pixels_agent_pos", observation_width=128, observation_height=128,
        init_states=True, episode_index=int(protocol["init_state_index"]), n_envs=1,
        num_steps_wait=int(protocol["physics_settle_steps"]),
        control_freq=int(protocol["control_frequency_hz"]), control_mode="relative", hard_reset=True,
    )
    started = time.perf_counter()
    try:
        observation, _ = env.reset(seed=int(protocol["evaluator_seed"]))
        pixels = observation.get("pixels", {})
        native = {name: np.asarray(pixels[key]) for name, key in protocol["observation_keys"].items()}
        direct = {
            name: np.asarray(env._env.sim.render(width=640, height=640, camera_name=camera))[::-1, ::-1].copy()
            for name, camera in protocol["simulator_camera_names"].items()
        }
        artifacts = {}
        for resolution, images in [("native128", native), ("direct640", direct)]:
            for name, array in images.items():
                expected = (128, 128, 3) if resolution == "native128" else (640, 640, 3)
                if array.shape != expected or array.dtype != np.uint8:
                    raise ValueError(f"invalid {resolution}/{name}: {array.shape} {array.dtype}")
                path = output / f"{name}_{resolution}.png"
                Image.fromarray(array, mode="RGB").save(path)
                artifacts[f"{name}_{resolution}"] = {
                    "path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                    "sha256": sha256(path), "stats": image_stats(array),
                }
        native_mae = float(np.mean(np.abs(native["agentview"].astype(np.int16) - native["eye_in_hand"].astype(np.int16))))
        direct_mae = float(np.mean(np.abs(direct["agentview"].astype(np.int16) - direct["eye_in_hand"].astype(np.int16))))
        checks = {
            "both_observation_keys_present": set(protocol["observation_keys"].values()).issubset(pixels),
            "native_images_nonconstant": all(array.std() > 1.0 for array in native.values()),
            "direct_images_nonconstant": all(array.std() > 1.0 for array in direct.values()),
            "native_viewpoints_not_identical": native_mae > 1.0,
            "direct_viewpoints_not_identical": direct_mae > 1.0,
        }
        result = {
            "classification": "PASS" if all(checks.values()) else "FAIL",
            "experiment_id": config["experiment_id"], "execution_pass": all(checks.values()),
            "checks": checks, "environment_resets": 1, "environment_steps": 0, "model_predictions": 0,
            "native_cross_view_mae": native_mae, "direct_cross_view_mae": direct_mae,
            "elapsed_s": time.perf_counter() - started, "artifacts": artifacts,
            "config_sha256": sha256(CONFIG), "boundary": config["boundary"],
        }
        result_path.write_text(yaml.safe_dump(result, sort_keys=False))
        print(json.dumps(result, indent=2))
        if result["classification"] != "PASS":
            raise SystemExit(1)
    finally:
        env.close()
        print(json.dumps({"cleanup": "env_closed", "environment_steps": 0, "model_predictions": 0}), flush=True)


if __name__ == "__main__":
    main()
