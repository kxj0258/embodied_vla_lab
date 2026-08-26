#!/usr/bin/env python
"""Strict offline preflight for the Phase 1C PushT Diffusion Policy baseline."""

from __future__ import annotations

import argparse
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def payload_manifest(root: Path) -> dict[str, int]:
    return dict(sorted(
        (str(path.relative_to(root)), path.stat().st_size)
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).parts[0] != ".cache"
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/diffusion_pusht_phase1c_baseline_30k.yaml"))
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/train/act_pusht_phase1b_deterministic_30k.yaml"))
    parser.add_argument("--dataset-contract", type=Path, default=Path("configs/dataset/pusht_v3.yaml"))
    args = parser.parse_args()

    required = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled"}
    wrong = {key: os.environ.get(key) for key, expected in required.items() if os.environ.get(key) != expected}
    if wrong:
        raise SystemExit(f"FAIL: strict offline flags missing or wrong: {wrong}")

    candidate = load_yaml(args.config.resolve())
    baseline = load_yaml(args.baseline_config.resolve())
    contract = load_yaml(args.dataset_contract.resolve())

    candidate_without_policy = dict(candidate)
    baseline_without_policy = dict(baseline)
    candidate_without_policy["policy"] = "<DIFFUSION_POLICY_BLOCK>"
    baseline_without_policy["policy"] = "<ACT_POLICY_BLOCK>"
    changed = {
        key for key in sorted(candidate_without_policy.keys() | baseline_without_policy.keys())
        if candidate_without_policy.get(key, "<MISSING>") != baseline_without_policy.get(key, "<MISSING>")
    }
    assert changed == {"ema", "job_name", "output_dir", "policy"}, changed
    for key in baseline["dataset"]:
        assert candidate["dataset"][key] == baseline["dataset"][key], key

    policy = candidate["policy"]
    assert policy["type"] == "diffusion"
    assert policy["n_obs_steps"] == 2 and policy["horizon"] == 48 and policy["n_action_steps"] == 10
    assert policy["horizon"] % (2 ** len(policy["down_dims"])) == 0
    assert policy["drop_n_last_frames"] == policy["horizon"] - policy["n_action_steps"] - policy["n_obs_steps"] + 1
    assert policy["pretrained_backbone_weights"] is None and policy["crop_is_random"] is False
    assert policy["num_train_timesteps"] == policy["num_inference_steps"] == 100
    assert candidate["steps"] == 30000 and candidate["batch_size"] == 8
    assert candidate["eval_steps"] == 5000 and candidate["save_freq"] == 10000
    assert candidate["accelerator"]["mixed_precision"] == "bf16"
    assert candidate["seed"] == 20260825 and candidate["cudnn_deterministic"] is True
    assert candidate["ema"] == {"enable": True, "use_for_eval": True}
    assert candidate["wandb"] == {"enable": False, "mode": "disabled"}
    assert candidate["env_eval_freq"] == 0 and candidate["eval"]["recording"] is False

    root = Path(candidate["dataset"]["root"])
    before = payload_manifest(root)
    assert len(before) == int(contract["expected"]["payload_files"])
    assert sum(before.values()) == int(contract["expected"]["payload_bytes"])
    output_dir = Path(candidate["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Candidate output directory already exists: {output_dir}")
    assert not torch.cuda.is_initialized()

    with tempfile.TemporaryDirectory(prefix="evla-phase1c-diffusion-audit-", dir="/tmp") as cache:
        os.environ["HF_DATASETS_CACHE"] = cache
        import draccus
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.datasets.factory import make_train_eval_datasets
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

        cfg = draccus.parse(config_class=TrainPipelineConfig, config_path=args.config.resolve(), args=[])
        cfg.validate()
        assert isinstance(cfg.policy, DiffusionConfig)
        optimizer = cfg.policy.get_optimizer_preset()
        scheduler = cfg.policy.get_scheduler_preset()
        assert optimizer.lr == 1e-4 and optimizer.betas == (0.95, 0.999)
        assert optimizer.weight_decay == 1e-6
        assert scheduler.name == "cosine" and scheduler.num_warmup_steps == 500
        train_dataset, eval_dataset = make_train_eval_datasets(cfg)
        assert eval_dataset is not None
        assert len(train_dataset.episodes) == 185 and train_dataset.num_frames == 23043
        assert len(eval_dataset.episodes) == 21 and eval_dataset.num_frames == 2607
        assert eval_dataset.episodes == contract["split"]["validation_episodes"]
        assert train_dataset.revision == eval_dataset.revision == contract["revision"]

    assert payload_manifest(root) == before and not output_dir.exists()
    assert not torch.cuda.is_initialized()

    print("PHASE1C_DIFFUSION_BASELINE_30K_CONFIG")
    print("static config contract: PASS")
    print("dataset contract: PASS (185/21 episodes; 23043/2607 frames)")
    print("registered model differences: ACT -> Diffusion; n_obs_steps=2; horizon=48; DDPM 100/100; EMA enabled")
    print("fixed controls: dataset revision, split, batch=8, BF16, execute=10, no pretrained backbone, seed, 5k eval, 10k checkpoints")

    if importlib.util.find_spec("diffusers") is None:
        raise SystemExit("PHASE 1C DIFFUSION BASELINE PREFLIGHT: FAIL (missing required package: diffusers)")
    print("PHASE 1C DIFFUSION BASELINE PREFLIGHT: PASS")
    print("No model, CUDA context, training, checkpoint, network, W&B, GUI, or upload was used.")


if __name__ == "__main__":
    main()
