#!/usr/bin/env python
"""Run one authorized PushT observation-to-action-to-step smoke on CUDA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import PushtEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint = args.checkpoint_dir.resolve()

    report = {
        "gate": "WP1.2_single_observation_action_step_smoke",
        "checkpoint_dir": str(checkpoint),
        "seed": args.seed,
        "device": args.device,
        "network_used": False,
        "training_executed": False,
        "episode_rollout_executed": False,
        "environment_steps": 0,
    }
    if not torch.cuda.is_available():
        report["status"] = "FAIL_CUDA_UNAVAILABLE"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 4

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = str(device)
    config.pretrained_path = str(checkpoint)
    config.pretrained_revision = None
    config.push_to_hub = False
    if hasattr(config, "use_amp"):
        config.use_amp = False

    load_started = time.perf_counter()
    policy = get_policy_class(config.type).from_pretrained(
        checkpoint, config=config, local_files_only=True
    )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        local_files_only=True,
    )
    torch.cuda.synchronize(device)
    report["policy_processor_load_s"] = time.perf_counter() - load_started

    env_values = yaml.safe_load(args.env_config.resolve().read_text())
    env_config = PushtEnv(**env_values)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_config, config)
    env = make_env(env_config, n_envs=1, use_async_envs=False)["pusht"][0]

    try:
        raw_observation, reset_info = env.reset(seed=args.seed)
        observation = preprocess_observation(raw_observation)
        report.update(
            {
                "raw_reset_info_keys": sorted(reset_info),
                "observation_image_shape": list(observation[OBS_IMAGE].shape),
                "observation_state_shape": list(observation[OBS_STATE].shape),
                "observation_image_finite": bool(torch.isfinite(observation[OBS_IMAGE]).all()),
                "observation_state_finite": bool(torch.isfinite(observation[OBS_STATE]).all()),
                "observation_image_min": float(observation[OBS_IMAGE].min()),
                "observation_image_max": float(observation[OBS_IMAGE].max()),
            }
        )
        observation["task"] = [""]
        observation = env_preprocessor(observation)
        processed = preprocessor(observation)
        policy.reset()

        torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            action = policy.select_action(processed)
        torch.cuda.synchronize(device)
        report["policy_select_action_s"] = time.perf_counter() - inference_started
        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]
        action_numpy = action.detach().cpu().numpy()

        report.update(
            {
                "action_shape": list(action_numpy.shape),
                "action_dtype": str(action_numpy.dtype),
                "action_values": action_numpy.tolist(),
                "action_finite": bool(np.isfinite(action_numpy).all()),
                "action_min": float(action_numpy.min()),
                "action_max": float(action_numpy.max()),
                "action_space_low": np.asarray(env.single_action_space.low).tolist(),
                "action_space_high": np.asarray(env.single_action_space.high).tolist(),
            }
        )

        _, reward, terminated, truncated, step_info = env.step(action_numpy)
        report.update(
            {
                "environment_steps": 1,
                "reward": np.asarray(reward).tolist(),
                "terminated": np.asarray(terminated).tolist(),
                "truncated": np.asarray(truncated).tolist(),
                "step_info_keys": sorted(step_info),
                "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(device),
                "cuda_memory_reserved_bytes": torch.cuda.memory_reserved(device),
                "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
    finally:
        env.close()

    low = np.asarray(report["action_space_low"])
    high = np.asarray(report["action_space_high"])
    values = np.asarray(report["action_values"])
    action_in_bounds = bool(np.all(values >= low) and np.all(values <= high))
    report["action_in_environment_bounds"] = action_in_bounds
    passed = (
        report["observation_image_shape"] == [1, 3, 96, 96]
        and report["observation_state_shape"] == [1, 2]
        and report["action_shape"] == [1, 2]
        and report["observation_image_finite"]
        and report["observation_state_finite"]
        and report["action_finite"]
        and action_in_bounds
        and report["environment_steps"] == 1
    )
    report["status"] = "PASS_SINGLE_OBSERVATION_ACTION_STEP" if passed else "FAIL_SMOKE_CONTRACT"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
