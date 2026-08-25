#!/usr/bin/env python
"""Strictly offline audit for the Phase 1B formal 100k ACT PushT config."""

from __future__ import annotations

import argparse
import copy
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml


ALLOWED_CHANGES = {
    "output_dir",
    "job_name",
    "steps",
    "log_freq",
    "eval_steps",
    "max_eval_samples",
    "save_checkpoint",
    "save_freq",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train/act_pusht_phase1b_formal_100k.yaml")
    )
    parser.add_argument(
        "--resource-gate-config",
        type=Path,
        default=Path("configs/train/act_pusht_phase1b_fullsplit_batch8_100step.yaml"),
    )
    parser.add_argument(
        "--dataset-contract", type=Path, default=Path("configs/dataset/pusht_v3.yaml")
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def payload_manifest(root: Path) -> dict[str, int]:
    result = {}
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root).parts[0] != ".cache":
            result[str(path.relative_to(root))] = path.stat().st_size
    return dict(sorted(result.items()))


def main() -> None:
    args = parse_args()
    required = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    wrong = {key: os.environ.get(key) for key, value in required.items() if os.environ.get(key) != value}
    if wrong:
        raise SystemExit(f"Refusing to run without strict offline flags: {wrong}")

    config_path = args.config.resolve()
    formal = load_yaml(config_path)
    resource_gate = load_yaml(args.resource_gate_config.resolve())
    contract = load_yaml(args.dataset_contract.resolve())

    formal_invariants = copy.deepcopy(formal)
    gate_invariants = copy.deepcopy(resource_gate)
    for key in ALLOWED_CHANGES:
        formal_invariants.pop(key)
        gate_invariants.pop(key)
    assert formal_invariants == gate_invariants, "Formal config drifted outside allowed fields"

    expected_changes = {
        "steps": 100000,
        "batch_size": 8,
        "log_freq": 200,
        "eval_steps": 5000,
        "max_eval_samples": 0,
        "save_checkpoint": True,
        "save_freq": 25000,
    }
    for key, value in expected_changes.items():
        assert formal[key] == value, (key, formal[key], value)
    assert formal["output_dir"] == "outputs/train/phase1b_act_pusht_formal_100k_bf16"
    assert formal["policy"]["push_to_hub"] is False
    assert formal["save_checkpoint_to_hub"] is False
    assert formal["wandb"]["enable"] is False
    assert formal["env_eval_freq"] == 0 and formal.get("env") is None
    assert formal["eval"]["recording"] is False
    assert formal["policy"]["pretrained_backbone_weights"] is None
    assert formal["ema"]["enable"] is False

    root = Path(formal["dataset"]["root"])
    before_manifest = payload_manifest(root)
    assert len(before_manifest) == int(contract["expected"]["payload_files"])
    assert sum(before_manifest.values()) == int(contract["expected"]["payload_bytes"])
    output_dir = Path(formal["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Formal output directory already exists: {output_dir}")
    assert not torch.cuda.is_initialized()

    temporary_cache_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="evla-phase1b-formal-audit-", dir="/tmp") as cache:
        temporary_cache_path = Path(cache)
        os.environ["HF_DATASETS_CACHE"] = cache
        import draccus

        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.datasets.factory import make_train_eval_datasets
        from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401

        cfg = draccus.parse(config_class=TrainPipelineConfig, config_path=config_path, args=[])
        cfg.validate()
        assert cfg.steps == 100000 and cfg.batch_size == 8
        assert cfg.accelerator.mixed_precision == "bf16"
        assert cfg.accelerator.gradient_accumulation.steps == 1
        assert cfg.eval_steps == 5000 and cfg.max_eval_samples == 0
        assert cfg.save_checkpoint is True and cfg.save_freq == 25000
        assert cfg.save_checkpoint_to_hub is False and cfg.wandb.enable is False

        train_dataset, eval_dataset = make_train_eval_datasets(cfg)
        assert eval_dataset is not None
        assert len(train_dataset.episodes) == 185 and train_dataset.num_frames == 23043
        assert len(eval_dataset.episodes) == 21 and eval_dataset.num_frames == 2607
        assert eval_dataset.episodes == contract["split"]["validation_episodes"]
        assert train_dataset.revision == eval_dataset.revision == contract["revision"]

        print("PHASE1B_FORMAL_CONFIG")
        print(f"config: {config_path}")
        print(f"output dir (not created): {cfg.output_dir}")
        print(f"train / validation: {len(train_dataset.episodes)} eps, {train_dataset.num_frames} frames / {len(eval_dataset.episodes)} eps, {eval_dataset.num_frames} frames")
        print(f"steps / batch / effective epochs: {cfg.steps} / {cfg.batch_size} / {cfg.steps * cfg.batch_size / train_dataset.num_frames:.4f}")
        print(f"log / eval / save cadence: {cfg.log_freq} / {cfg.eval_steps} / {cfg.save_freq}")
        print("full validation samples per eval: 2607")
        print("checkpoint steps: [25000, 50000, 75000, 100000]")
        print("formal config differs from the passed resource gate only in the allowed fields: True")

    assert temporary_cache_path is not None and not temporary_cache_path.exists()
    assert payload_manifest(root) == before_manifest
    assert not output_dir.exists()
    assert not torch.cuda.is_initialized()
    print("\nPHASE 1B ACT FORMAL CONFIG/DATASET PREFLIGHT: PASS")
    print("No model, CUDA context, training, checkpoint, network, W&B, GUI, or upload was used.")


if __name__ == "__main__":
    main()
