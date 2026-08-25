#!/usr/bin/env python
"""Strictly offline audit for the Phase 1B ACT smoke configuration.

This script parses and validates the training config and constructs only the train/eval
datasets. It does not instantiate ACT, initialize CUDA, create the training output directory,
run an optimizer step, save a checkpoint, contact W&B, or upload anything.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/act_pusht_phase1b_smoke.yaml"),
    )
    parser.add_argument(
        "--dataset-contract",
        type=Path,
        default=Path("configs/dataset/pusht_v3.yaml"),
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def payload_manifest(root: Path) -> dict[str, int]:
    manifest: dict[str, int] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        manifest[str(relative)] = path.stat().st_size
    return dict(sorted(manifest.items()))


def require_offline_environment() -> None:
    required = {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    wrong = {key: os.environ.get(key) for key, expected in required.items() if os.environ.get(key) != expected}
    if wrong:
        raise SystemExit(f"Refusing to run without strict offline flags: {wrong}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    contract_path = args.dataset_contract.resolve()
    raw = load_yaml(config_path)
    contract = load_yaml(contract_path)

    require_offline_environment()

    validation_ids = [int(value) for value in contract["split"]["validation_episodes"]]
    total_episodes = int(contract["expected"]["total_episodes"])
    validation_set = set(validation_ids)
    training_ids = [episode for episode in range(total_episodes) if episode not in validation_set]
    ordered_ids = training_ids + validation_ids

    dataset_raw = raw["dataset"]
    policy_raw = raw["policy"]
    assert dataset_raw["repo_id"] == contract["repo_id"]
    assert dataset_raw["revision"] == contract["revision"]
    assert Path(dataset_raw["root"]) == Path(contract["root"])
    assert dataset_raw["video_backend"] == contract["video_backend"] == "pyav"
    assert dataset_raw["episodes"] == ordered_ids
    assert math.ceil(total_episodes * float(dataset_raw["eval_split"])) == len(validation_ids)
    assert dataset_raw["use_imagenet_stats"] is False
    assert policy_raw["type"] == "act"
    assert policy_raw["chunk_size"] == contract["first_training_candidate"]["chunk_size"]
    assert policy_raw["n_action_steps"] == contract["first_training_candidate"]["n_action_steps"]
    assert policy_raw["pretrained_backbone_weights"] is None

    safety_checks = {
        "wandb disabled": raw["wandb"]["enable"] is False and raw["wandb"]["mode"] == "disabled",
        "policy Hub push disabled": policy_raw["push_to_hub"] is False,
        "checkpoint Hub push disabled": raw["save_checkpoint_to_hub"] is False,
        "environment evaluation disabled": raw["env_eval_freq"] == 0 and raw.get("env") is None,
        "eval recording disabled": raw["eval"]["recording"] is False,
        "checkpoint writing disabled": raw["save_checkpoint"] is False,
        "pretrained download disabled": policy_raw["pretrained_backbone_weights"] is None,
    }
    assert all(safety_checks.values()), safety_checks

    root = Path(dataset_raw["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Pinned local dataset root is missing: {root}")
    before_manifest = payload_manifest(root)
    expected = contract["expected"]
    assert len(before_manifest) == int(expected["payload_files"])
    assert sum(before_manifest.values()) == int(expected["payload_bytes"])

    output_dir = Path(raw["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Smoke output directory already exists: {output_dir}")
    assert not torch.cuda.is_initialized()

    temporary_cache_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="evla-phase1b-config-audit-", dir="/tmp") as temporary_cache:
        temporary_cache_path = Path(temporary_cache)
        os.environ["HF_DATASETS_CACHE"] = temporary_cache

        import draccus

        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.datasets.factory import make_train_eval_datasets
        from lerobot.policies.act.configuration_act import ACTConfig

        cfg = draccus.parse(config_class=TrainPipelineConfig, config_path=config_path, args=[])
        cfg.validate()

        assert isinstance(cfg.policy, ACTConfig)
        assert cfg.policy.push_to_hub is False
        assert cfg.policy.pretrained_backbone_weights is None
        assert cfg.policy.chunk_size == 50
        assert cfg.policy.n_action_steps == 10
        assert cfg.accelerator.mixed_precision == "bf16"
        assert cfg.accelerator.gradient_accumulation.steps == 1
        assert cfg.batch_size == 1
        assert cfg.steps == 2
        assert cfg.num_workers == 0
        assert cfg.eval_steps == 0
        assert cfg.env_eval_freq == 0
        assert cfg.save_checkpoint is False
        assert cfg.wandb.enable is False
        assert cfg.save_checkpoint_to_hub is False

        train_dataset, eval_dataset = make_train_eval_datasets(cfg)
        assert eval_dataset is not None
        assert train_dataset.episodes == training_ids
        assert eval_dataset.episodes == validation_ids
        assert train_dataset.revision == contract["revision"]
        assert eval_dataset.revision == contract["revision"]
        assert Path(train_dataset.root) == root
        assert Path(eval_dataset.root) == root

        published_stats = json.loads((root / "meta/stats.json").read_text())
        for dataset in (train_dataset, eval_dataset):
            for stat_name in ("mean", "std"):
                # Dataset metadata may expose stats as NumPy arrays or tensors depending on
                # the factory path. Normalize both representations before comparing values.
                actual = torch.as_tensor(dataset.meta.stats["observation.image"][stat_name]).cpu()
                expected_stat = torch.tensor(
                    published_stats["observation.image"][stat_name], dtype=actual.dtype
                )
                assert torch.equal(actual, expected_stat)

        print("PHASE1B_CONFIG")
        print(f"config: {config_path}")
        print(f"output dir (not created): {cfg.output_dir}")
        print(f"steps / batch / grad accumulation: {cfg.steps} / {cfg.batch_size} / {cfg.accelerator.gradient_accumulation.steps}")
        print(f"mixed precision: {cfg.accelerator.mixed_precision}")
        print(f"chunk / action steps: {cfg.policy.chunk_size} / {cfg.policy.n_action_steps}")
        print(f"optimizer: {type(cfg.optimizer).__name__}")
        print(f"scheduler: {cfg.scheduler}")

        print("\nDATASET_SPLIT")
        print(f"train episodes: {len(train_dataset.episodes)}")
        print(f"validation episodes: {eval_dataset.episodes}")
        print(f"dataset revision: {train_dataset.revision}")
        print(f"video backend: {dataset_raw['video_backend']}")
        print("snapshot image statistics preserved: True")

        print("\nSAFETY_GATES")
        for name, passed in safety_checks.items():
            print(f"{name}: {passed}")

    assert temporary_cache_path is not None and not temporary_cache_path.exists()
    assert payload_manifest(root) == before_manifest
    assert not output_dir.exists()
    assert not torch.cuda.is_initialized()

    print("\nPHASE 1B ACT CONFIG/DATASET PREFLIGHT: PASS")
    print("Dataset payload remained unchanged and the temporary datasets cache was removed.")
    print("No model, CUDA context, training, checkpoint, network, W&B, GUI, or upload was used.")


if __name__ == "__main__":
    main()
