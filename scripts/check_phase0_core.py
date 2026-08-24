#!/usr/bin/env python
"""Host-side Phase 0 core smoke test.

This script performs no training, dataset/model download, file write, or GUI
rendering. It verifies the pinned Python package contract, CUDA/BF16 execution,
the project PushT observation contract, and an untrained ACT image-to-action
forward pass. Randomly initialized actions are interface checks only.
"""

from __future__ import annotations

import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.dont_write_bytecode = True

import cv2
import torch
import yaml

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import PushtEnv
from lerobot.envs.factory import make_env
from lerobot.envs.utils import env_to_policy_features, preprocess_observation
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUSHT_CONFIG = PROJECT_ROOT / "configs" / "env" / "pusht_96.yaml"

EXPECTED_PACKAGES = {
    "lerobot": "0.6.2",
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "gym-pusht": "0.1.6",
    "gymnasium": "1.3.0",
    "opencv-python-headless": "4.13.0.92",
}


def check_packages() -> None:
    print("\n[1/4] Package contract")
    for package, expected in EXPECTED_PACKAGES.items():
        actual = version(package)
        print(f"{package}: {actual}")
        assert actual == expected, f"{package}: expected {expected}, got {actual}"

    try:
        version("opencv-python")
    except PackageNotFoundError:
        pass
    else:
        raise RuntimeError(
            "opencv-python and opencv-python-headless must not coexist in evla-core"
        )

    print(f"cv2 import: {cv2.__version__}")
    print("Package contract: PASS")


def initialize_cuda() -> tuple[int, torch.device]:
    print("\n[2/4] CUDA and BF16")
    assert torch.cuda.is_available(), "PyTorch cannot access CUDA"

    device_index = 0
    torch.cuda.set_device(device_index)
    device = torch.device("cuda:0")

    context_probe = torch.empty(1, device=device)
    del context_probe
    torch.cuda.synchronize(device_index)

    capability = torch.cuda.get_device_capability(device_index)
    arch_list = torch.cuda.get_arch_list()

    print(f"device: {torch.cuda.get_device_name(device_index)}")
    print(f"compiled CUDA: {torch.version.cuda}")
    print(f"compute capability: {capability}")
    print(f"architecture list: {arch_list}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

    assert torch.version.cuda == "12.8"
    assert capability == (12, 0)
    assert "sm_120" in arch_list
    assert torch.cuda.is_bf16_supported()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    a = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16)
    b = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16)
    c = a @ b
    torch.cuda.synchronize(device_index)
    assert c.dtype is torch.bfloat16
    assert torch.isfinite(c).all()

    del a, b, c
    torch.cuda.empty_cache()
    print("CUDA/BF16: PASS")
    return device_index, device


def load_pusht_config() -> PushtEnv:
    with PUSHT_CONFIG.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    if not isinstance(values, dict):
        raise TypeError(f"Expected a mapping in {PUSHT_CONFIG}")
    return PushtEnv(**values)


def check_pusht() -> tuple[object, dict[str, torch.Tensor]]:
    print("\n[3/4] LeRobot PushT wrapper")
    env_cfg = load_pusht_config()
    features = env_to_policy_features(env_cfg)

    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs["pusht"][0]
    raw_observation, _ = env.reset(seed=0)
    observation = preprocess_observation(raw_observation)

    declared_image_shape = features[OBS_IMAGE].shape
    runtime_image_shape = tuple(observation[OBS_IMAGE].shape[1:])
    runtime_state_shape = tuple(observation[OBS_STATE].shape[1:])

    print(f"config: {PUSHT_CONFIG}")
    print(f"declared image: {declared_image_shape}")
    print(f"runtime image: {runtime_image_shape}")
    print(f"runtime state: {runtime_state_shape}")
    print(f"action feature: {features[ACTION].shape}")

    assert declared_image_shape == (3, 96, 96)
    assert runtime_image_shape == (3, 96, 96)
    assert runtime_state_shape == (2,)
    assert features[ACTION].shape == (2,)
    assert observation[OBS_IMAGE].dtype is torch.float32
    assert 0.0 <= float(observation[OBS_IMAGE].min())
    assert float(observation[OBS_IMAGE].max()) <= 1.0

    upstream_default_shape = env_to_policy_features(PushtEnv())[OBS_IMAGE].shape
    if upstream_default_shape != runtime_image_shape:
        print(
            "known upstream mismatch: default declaration "
            f"{upstream_default_shape}, gym-pusht runtime {runtime_image_shape}"
        )

    print("LeRobot PushT wrapper: PASS")
    return env, observation


def check_act(
    observation: dict[str, torch.Tensor],
    device_index: int,
    device: torch.device,
) -> None:
    print("\n[4/4] ACT image-to-action interface")
    image_shape = tuple(observation[OBS_IMAGE].shape[1:])
    state_shape = tuple(observation[OBS_STATE].shape[1:])

    config = ACTConfig(
        input_features={
            OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=state_shape),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
        device="cuda",
        chunk_size=100,
        n_action_steps=10,
        pretrained_backbone_weights=None,
    )
    config.validate_features()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    policy = ACTPolicy(config).to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    batch = {
        OBS_IMAGE: observation[OBS_IMAGE].to(device),
        OBS_STATE: observation[OBS_STATE].to(device),
    }

    torch.cuda.synchronize(device_index)
    start = time.perf_counter()
    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(batch)
    torch.cuda.synchronize(device_index)
    elapsed = time.perf_counter() - start

    print(f"parameters: {parameter_count / 1e6:.2f} M")
    print(f"action chunk: {tuple(action_chunk.shape)}, {action_chunk.dtype}")
    print(f"first random action: {action_chunk[:, 0].cpu().numpy()}")
    print(f"first-pass latency: {elapsed:.6f} s")
    print(
        "peak PyTorch allocation: "
        f"{torch.cuda.max_memory_allocated(device_index) / 1024**2:.2f} MiB"
    )

    assert 50_000_000 < parameter_count < 53_000_000
    assert tuple(action_chunk.shape) == (1, 100, 2)
    assert torch.isfinite(action_chunk).all()

    print("ACT image-to-action interface: PASS")
    print("Random initialization: action values have no control meaning.")


def main() -> None:
    check_packages()
    device_index, device = initialize_cuda()
    env, observation = check_pusht()
    try:
        check_act(observation, device_index, device)
    finally:
        env.close()
    print("\nPhase 0 core smoke test: PASS")


if __name__ == "__main__":
    main()
