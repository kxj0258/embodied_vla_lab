#!/usr/bin/env python
"""Offline fixed-dataset accuracy audit for the Phase 1B ACT overfit checkpoint.

This loads the local checkpoint and saved processors, predicts action chunks for every
frame in one train and one validation episode, and compares predictions with non-padding
dataset targets. It does not train, save a checkpoint, run an environment, use the network
or W&B, start a GUI, or upload anything.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors


TRAIN_EPISODE = 51
VALIDATION_EPISODE = 41
EXPECTED_FRAMES = 73
EXPECTED_CHUNK_SIZE = 50
EXPECTED_ACTION_STEPS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/train/phase1b_act_pusht_overfit_ep51_bf16/"
            "checkpoints/002000/pretrained_model"
        ),
    )
    parser.add_argument(
        "--dataset-contract",
        type=Path,
        default=Path("configs/dataset/pusht_v3.yaml"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_offline_environment() -> None:
    required = {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    wrong = {key: os.environ.get(key) for key, expected in required.items() if os.environ.get(key) != expected}
    if wrong:
        raise SystemExit(f"Refusing to run without strict offline flags: {wrong}")


def make_episode_dataset(contract: dict[str, Any], episode: int) -> LeRobotDataset:
    return LeRobotDataset(
        repo_id=contract["repo_id"],
        root=Path(contract["root"]),
        episodes=[episode],
        revision=contract["revision"],
        video_backend=contract["video_backend"],
        delta_timestamps={
            "action": [index / float(contract["fps"]) for index in range(EXPECTED_CHUNK_SIZE)]
        },
    )


def audit_episode(
    name: str,
    dataset: LeRobotDataset,
    policy: torch.nn.Module,
    preprocessor,
    postprocessor,
    batch_size: int,
) -> dict[str, float | int]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    full_sum = 0.0
    full_count = 0
    executed_sum = 0.0
    executed_count = 0
    first_sum = 0.0
    first_count = 0
    predicted_min = float("inf")
    predicted_max = float("-inf")
    outside_range = 0
    predicted_values = 0

    with torch.inference_mode():
        for batch in loader:
            observation = {key: value for key, value in batch.items() if key.startswith("observation.")}
            processed = preprocessor(observation)
            normalized_chunk = policy.predict_action_chunk(processed)
            prediction = postprocessor(normalized_chunk).cpu()
            target = batch["action"].cpu()
            pad = batch["action_is_pad"].cpu()

            assert prediction.shape == target.shape
            assert prediction.shape[1:] == (EXPECTED_CHUNK_SIZE, 2)
            assert pad.shape == prediction.shape[:2]
            assert torch.isfinite(prediction).all()

            absolute_error = (prediction - target).abs()
            valid = (~pad).unsqueeze(-1).expand_as(absolute_error)
            full_sum += absolute_error[valid].sum().item()
            full_count += int(valid.sum().item())

            executed_valid = valid[:, :EXPECTED_ACTION_STEPS]
            executed_sum += absolute_error[:, :EXPECTED_ACTION_STEPS][executed_valid].sum().item()
            executed_count += int(executed_valid.sum().item())

            first_valid = valid[:, 0]
            first_sum += absolute_error[:, 0][first_valid].sum().item()
            first_count += int(first_valid.sum().item())

            predicted_min = min(predicted_min, prediction.min().item())
            predicted_max = max(predicted_max, prediction.max().item())
            outside_range += int(((prediction < 0) | (prediction > 512)).sum().item())
            predicted_values += prediction.numel()

    result: dict[str, float | int] = {
        "frames": len(dataset),
        "full_chunk_valid_values": full_count,
        "full_chunk_mae": full_sum / full_count,
        "first_10_valid_values": executed_count,
        "first_10_mae": executed_sum / executed_count,
        "first_action_values": first_count,
        "first_action_mae": first_sum / first_count,
        "prediction_min": predicted_min,
        "prediction_max": predicted_max,
        "outside_0_512_values": outside_range,
        "predicted_values": predicted_values,
    }
    print(f"{name.upper()}_EPISODE_ACCURACY")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    return result


def main() -> None:
    args = parse_args()
    require_offline_environment()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this checkpoint accuracy audit")

    checkpoint = args.checkpoint.resolve()
    contract = load_yaml(args.dataset_contract.resolve())
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    validation_ids = {int(value) for value in contract["split"]["validation_episodes"]}
    assert TRAIN_EPISODE not in validation_ids
    assert VALIDATION_EPISODE in validation_ids
    assert contract["video_backend"] == "pyav"
    assert contract["first_training_candidate"]["chunk_size"] == EXPECTED_CHUNK_SIZE
    assert contract["first_training_candidate"]["n_action_steps"] == EXPECTED_ACTION_STEPS

    train_dataset = make_episode_dataset(contract, TRAIN_EPISODE)
    validation_dataset = make_episode_dataset(contract, VALIDATION_EPISODE)
    assert len(train_dataset) == len(validation_dataset) == EXPECTED_FRAMES

    cfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg.device = "cuda"
    cfg.pretrained_path = str(checkpoint)
    cfg.pretrained_revision = None
    cfg.push_to_hub = False
    assert cfg.chunk_size == EXPECTED_CHUNK_SIZE
    assert cfg.n_action_steps == EXPECTED_ACTION_STEPS

    policy = make_policy(cfg=cfg, ds_meta=train_dataset.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    torch.cuda.reset_peak_memory_stats()
    train_result = audit_episode(
        "train", train_dataset, policy, preprocessor, postprocessor, args.batch_size
    )
    validation_result = audit_episode(
        "validation", validation_dataset, policy, preprocessor, postprocessor, args.batch_size
    )
    torch.cuda.synchronize()

    print("\nCHECKPOINT_ACCURACY_SUMMARY")
    print(f"checkpoint: {checkpoint}")
    print(f"model_sha256: {sha256(model_path)}")
    print(f"device: {next(policy.parameters()).device}")
    print(f"batch_size: {args.batch_size}")
    print(f"train_to_validation_full_chunk_mae_ratio: {train_result['full_chunk_mae'] / validation_result['full_chunk_mae']:.6f}")
    print(f"peak_pytorch_allocated_mib: {torch.cuda.max_memory_allocated() / 1024**2:.2f}")
    print("PHASE 1B ACT CHECKPOINT FIXED-DATA ACCURACY: PASS")
    print("No training, environment rollout, checkpoint write, network, W&B, GUI, or upload was used.")


if __name__ == "__main__":
    main()
