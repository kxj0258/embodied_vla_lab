#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/presentation/phase4d_wrist_fullval_summary_v1_20260828"
PNG = OUT / "phase4d_wrist_base_vs_qlora_fullval889.png"
SVG = OUT / "phase4d_wrist_base_vs_qlora_fullval889.svg"
MANIFEST = ROOT / "results/phase4d_wrist_fullval_summary_v1_manifest.yaml"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists() or MANIFEST.exists():
        raise SystemExit("REFUSING_TO_OVERWRITE: Phase4D summary target exists")
    OUT.mkdir(parents=True)
    labels = ["Mean loss\n(lower better)", "Token accuracy\n(higher better)", "Continuous L1\n(lower better)"]
    base = [16.6833914176, 13.0805077937, 0.2879627442]
    adapter = [9.0479311779, 20.8098987627, 0.1945887016]
    changes = ["−45.77%", "+7.73 pp", "−32.43%"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    colors = ["#64748b", "#2563eb"]
    for ax, title, b, a, delta in zip(axes, labels, base, adapter, changes):
        bars = ax.bar(["Frozen base", "Wrist QLoRA\nstep-20"], [b, a], color=colors, width=.62)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, [b, a]):
            fmt = f"{value:.2f}%" if "accuracy" in title.lower() else f"{value:.3f}"
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), fmt, ha="center", va="bottom", fontsize=10)
        ax.text(.5, .91, delta, transform=ax.transAxes, ha="center", color="#166534", weight="bold")
        ax.set_ylim(0, max(b, a) * 1.26)
    fig.suptitle("Native wrist fullval889: frozen OpenVLA base vs rank-8 QLoRA", fontsize=15, weight="bold")
    fig.text(.5, .015, "889 paired frames · 6,223 action tokens · all 10 episodes improved on all three metrics\nTeacher-forced offline metrics — not environment success or reward", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, .10, 1, .91])
    fig.savefig(PNG, dpi=180)
    fig.savefig(SVG)
    plt.close(fig)
    manifest = f'''asset: phase4d_wrist_fullval_summary_v1\nclassification: PASS\nsource: results/phase4d_openvla_wrist_step20_vs_base_fullval889_paired.yaml\nartifacts:\n  png: {{path: {PNG.relative_to(ROOT)}, bytes: {PNG.stat().st_size}, sha256: {sha(PNG)}}}\n  svg: {{path: {SVG.relative_to(ROOT)}, bytes: {SVG.stat().st_size}, sha256: {sha(SVG)}}}\nvisual_qa: PENDING\nboundary: CPU-only visualization of offline paired metrics; not environment success or reward.\n'''
    MANIFEST.write_text(manifest)
    print(json.dumps({"classification": "PASS", "png": str(PNG), "manifest": str(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
