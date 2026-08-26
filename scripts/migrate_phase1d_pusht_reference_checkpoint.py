#!/usr/bin/env python
"""Create an offline current-format bundle from the frozen legacy PushT checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.migrate_policy_normalization import (
    clean_state_dict,
    extract_normalization_stats,
    remove_normalization_layers,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    partial = output.with_name(f"{output.name}.partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")

    required = [source / "config.json", source / "model.safetensors"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source files: {missing}")

    started = time.perf_counter()
    state = load_file(source / "model.safetensors", device="cpu")
    stats = extract_normalization_stats(state)
    core = clean_state_dict(remove_normalization_layers(state), remove_str="._orig_mod")

    config = PreTrainedConfig.from_pretrained(source, local_files_only=True)
    config.device = "cpu"
    config.pretrained_path = None
    policy = get_policy_class(config.type)(config)
    incompatible = policy.load_state_dict(core, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load mismatch: {incompatible}")

    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=config, dataset_stats=stats)
    partial.mkdir(parents=True)
    preprocessor.save_pretrained(partial)
    postprocessor.save_pretrained(partial)
    policy.save_pretrained(partial)

    for name in ("README.md", "train_config.json"):
        source_file = source / name
        if source_file.is_file():
            shutil.copy2(source_file, partial / name)

    report = {
        "status": "PASS_MIGRATED_CURRENT_FORMAT_BUNDLE",
        "source_dir": str(source),
        "output_dir": str(output),
        "network_used": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "strict_load": True,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "archived_tensor_count": len(state),
        "core_tensor_count": len(core),
        "removed_normalization_tensor_count": len(state) - len(core),
        "normalization_features": sorted(stats),
        "forward_executed": False,
        "environment_constructed": False,
        "elapsed_s": time.perf_counter() - started,
    }
    (partial / "phase1d_migration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    os.rename(partial, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
