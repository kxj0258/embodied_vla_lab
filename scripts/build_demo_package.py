#!/usr/bin/env python3
"""Assemble a lightweight interview demo package from existing artifacts.

The package uses symlinks for large media, so it does not duplicate videos or
model files.  It is deliberately read-only with respect to source artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link(package: Path, label: str, source: Path, entries: list[dict]) -> None:
    target = package / label
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        entries.append({"label": label, "source": str(source), "status": "missing"})
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.symlink_to(source)
    entries.append({"label": label, "source": str(source), "status": "linked", "bytes": source.stat().st_size})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo/embodied_vla_demo_v1")
    args = parser.parse_args()
    package = args.output if args.output.is_absolute() else ROOT / args.output
    if package.exists():
        raise FileExistsError(f"refusing to overwrite demo package: {package}")
    package.mkdir(parents=True)
    entries: list[dict] = []
    assets = {
        "videos/act_libero_success.mp4": ROOT / "outputs/presentation/p0_hd_act_ckpt100000_init05_v1_seed20260901/act_libero_hd640_init05_success.mp4",
        "videos/openvla_single_task_success.mp4": ROOT / "outputs/presentation/p0_hd_openvla_base_init05_v2_seed20260901/presentation_hd640.mp4",
        "videos/openvla_multitask_montage.mp4": ROOT / "outputs/presentation/p0_breadth_10task_montage_v1_20260828/openvla_10task_success_montage_hd.mp4",
        "videos/base_vs_qlora_speedup.mp4": ROOT / "outputs/presentation/p0_hd_comparisons_v1_20260828/base_vs_qlora_init40_typical_speedup_hd.mp4",
        "videos/base_vs_qlora_tail_failure.mp4": ROOT / "outputs/presentation/p0_hd_comparisons_v1_20260828/base_vs_qlora_init44_tail_failure_hd.mp4",
        "reports/failure_analysis.md": ROOT / "results/benchmark/failure_analysis_v1.md",
        "reports/benchmark_report.md": ROOT / "results/benchmark/report_v1/report.md",
        "reports/project_report.md": ROOT / "docs/project_report.md",
    }
    for label, source in assets.items():
        link(package, label, source, entries)
    result_sources = [
        ROOT / "results/benchmark/benchmark_act_libero_heldout45_v1/result.json",
        ROOT / "results/benchmark/benchmark_openvla_libero_heldout45_v1/result.json",
        ROOT / "results/benchmark/benchmark_openvla_libero_breadth50_v1/result.json",
        ROOT / "results/benchmark/benchmark_openvla_libero_breadth100_v1/result.json",
    ]
    (package / "results").mkdir()
    result_index = []
    for source in result_sources:
        if source.exists():
            destination = package / "results" / source.parent.name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source)
            result_index.append(str(source))
    (package / "asset_manifest.json").write_text(json.dumps({"schema_version": "evla.demo.v1", "assets": entries, "results": result_index}, indent=2) + "\n")
    (package / "README.md").write_text("""# Embodied VLA Lab — Interview Demo Package

This package is assembled from existing experiment artifacts. Large videos are
symlinks and are not duplicated. Open the videos in `videos/` first, then use
the reports in `reports/` to explain the evaluation boundaries.

## Recommended 3-minute flow

1. `videos/openvla_multitask_montage.mp4` — language-conditioned multi-task manipulation.
2. `videos/act_libero_success.mp4` — fast ACT single-task closed loop.
3. `videos/base_vs_qlora_speedup.mp4` — offline adaptation and efficiency comparison.
4. `reports/failure_analysis.md` — why multi-task success is 66/100 rather than universal success.

## Frozen evidence

- ACT single-task held-out: 45/45.
- OpenVLA unified single-task held-out: 44/45; one timeout at init state 30.
- OpenVLA breadth: 33/50 and 66/100.
- All multi-task failures in the 100-episode result are horizon timeouts.

The package does not claim official full LIBERO benchmark performance or real-robot transfer.
""")
    print(package)
    print(package / "README.md")
    print(package / "asset_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
