#!/usr/bin/env python
"""Strict offline preflight for the Route B deterministic ACT 30k ablation."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml


ALLOWED_DIFFS = {
    "job_name",
    "output_dir",
    "policy.use_vae",
    "save_freq",
    "steps",
}

EXPECTED_DIFFS = {
    "job_name": ("phase1b_act_pusht_formal_100k_bf16", "phase1b_act_pusht_deterministic_30k_bf16"),
    "output_dir": (
        "outputs/train/phase1b_act_pusht_formal_100k_bf16",
        "outputs/train/phase1b_act_pusht_deterministic_30k_bf16",
    ),
    "policy.use_vae": ("<MISSING>", False),
    "save_freq": (25000, 10000),
    "steps": (100000, 30000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/act_pusht_phase1b_deterministic_30k.yaml"),
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/train/act_pusht_phase1b_formal_100k.yaml"),
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


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(value[key], path))
        return result
    return {prefix: value}


def config_diffs(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    missing = "<MISSING>"
    left = flatten(baseline)
    right = flatten(candidate)
    return {
        key: (left.get(key, missing), right.get(key, missing))
        for key in sorted(left.keys() | right.keys())
        if left.get(key, missing) != right.get(key, missing)
    }


def payload_manifest(root: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root).parts[0] != ".cache":
            result[str(path.relative_to(root))] = path.stat().st_size
    return dict(sorted(result.items()))


def main() -> None:
    args = parse_args()
    required = {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
    }
    wrong = {key: os.environ.get(key) for key, value in required.items() if os.environ.get(key) != value}
    if wrong:
        raise SystemExit(f"Refusing to run without strict offline flags: {wrong}")

    config_path = args.config.resolve()
    baseline_path = args.baseline_config.resolve()
    candidate = load_yaml(config_path)
    baseline = load_yaml(baseline_path)
    contract = load_yaml(args.dataset_contract.resolve())

    diffs = config_diffs(baseline, candidate)
    assert set(diffs) == ALLOWED_DIFFS, ("unregistered or missing config differences", diffs)
    assert diffs == EXPECTED_DIFFS, ("pre-registered differences changed", diffs)

    assert candidate["policy"]["use_vae"] is False
    assert candidate["steps"] == 30000
    assert candidate["eval_steps"] == 5000 and candidate["max_eval_samples"] == 0
    assert candidate["save_checkpoint"] is True and candidate["save_freq"] == 10000
    assert candidate["batch_size"] == 8
    assert candidate["accelerator"]["mixed_precision"] == "bf16"
    assert candidate["accelerator"]["gradient_accumulation"]["steps"] == 1
    assert candidate["seed"] == 20260825 and candidate["cudnn_deterministic"] is True
    assert candidate["policy"]["chunk_size"] == 50
    assert candidate["policy"]["n_action_steps"] == 10
    assert candidate["policy"]["pretrained_backbone_weights"] is None
    assert candidate["policy"]["push_to_hub"] is False
    assert candidate["save_checkpoint_to_hub"] is False
    assert candidate["wandb"] == {"enable": False, "mode": "disabled"}
    assert candidate["env_eval_freq"] == 0 and candidate.get("env") is None
    assert candidate["eval"]["recording"] is False
    assert candidate["ema"]["enable"] is False

    root = Path(candidate["dataset"]["root"])
    before_manifest = payload_manifest(root)
    assert len(before_manifest) == int(contract["expected"]["payload_files"])
    assert sum(before_manifest.values()) == int(contract["expected"]["payload_bytes"])
    output_dir = Path(candidate["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Route B output directory already exists: {output_dir}")
    assert not torch.cuda.is_initialized()

    temporary_cache_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="evla-phase1b-deterministic-audit-", dir="/tmp") as cache:
        temporary_cache_path = Path(cache)
        os.environ["HF_DATASETS_CACHE"] = cache

        import draccus

        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.datasets.factory import make_train_eval_datasets
        from lerobot.policies.act.configuration_act import ACTConfig

        baseline_cfg = draccus.parse(
            config_class=TrainPipelineConfig, config_path=baseline_path, args=[]
        )
        cfg = draccus.parse(config_class=TrainPipelineConfig, config_path=config_path, args=[])
        # The completed baseline output intentionally exists. Parsing resolves its ACT
        # defaults without triggering the resume=false overwrite guard in validate().
        cfg.validate()
        assert isinstance(cfg.policy, ACTConfig)
        assert baseline_cfg.policy.use_vae is True and cfg.policy.use_vae is False
        assert cfg.steps == 30000 and cfg.batch_size == 8
        assert cfg.eval_steps == 5000 and cfg.max_eval_samples == 0
        assert cfg.save_checkpoint is True and cfg.save_freq == 10000
        assert cfg.accelerator.mixed_precision == "bf16"
        assert cfg.accelerator.gradient_accumulation.steps == 1

        train_dataset, eval_dataset = make_train_eval_datasets(cfg)
        assert eval_dataset is not None
        assert len(train_dataset.episodes) == 185 and train_dataset.num_frames == 23043
        assert len(eval_dataset.episodes) == 21 and eval_dataset.num_frames == 2607
        assert eval_dataset.episodes == contract["split"]["validation_episodes"]
        assert train_dataset.revision == eval_dataset.revision == contract["revision"]

        print("PHASE1B_ROUTE_B_DETERMINISTIC_30K_CONFIG")
        print(f"candidate: {config_path}")
        print(f"baseline: {baseline_path}")
        print("registered raw YAML differences:")
        for key, (old, new) in diffs.items():
            print(f"  {key}: {old!r} -> {new!r}")
        print("train / validation: 185 eps, 23043 frames / 21 eps, 2607 frames")
        print("steps / batch / effective epochs: 30000 / 8 / 10.4153")
        print("eval steps: [5000, 10000, 15000, 20000, 25000, 30000]")
        print("checkpoint steps: [10000, 20000, 30000]")
        print("core model change: policy.use_vae true -> false")

    assert temporary_cache_path is not None and not temporary_cache_path.exists()
    assert payload_manifest(root) == before_manifest
    assert not output_dir.exists()
    assert not torch.cuda.is_initialized()
    print("\nPHASE 1B ROUTE B DETERMINISTIC ACT 30K PREFLIGHT: PASS")
    print("No model, CUDA context, training, checkpoint, network, W&B, GUI, or upload was used.")


if __name__ == "__main__":
    main()
