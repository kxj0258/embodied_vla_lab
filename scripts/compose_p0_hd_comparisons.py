#!/usr/bin/env python3
"""Compose immutable P0-HD base-vs-QLoRA videos from completed clean captures."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = Path("/home/kxj/miniforge3/envs/evla-core/bin/ffmpeg")
FFPROBE = Path("/home/kxj/miniforge3/envs/evla-core/bin/ffprobe")
OUT = ROOT / "outputs/presentation/p0_hd_comparisons_v1_20260828"
RESULT = ROOT / "results/p0_hd_comparisons_v1_manifest.yaml"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probe(path: Path) -> dict:
    raw = subprocess.check_output(
        [str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames,duration", "-of", "json", str(path)],
        text=True,
    )
    return json.loads(raw)["streams"][0]


def compose(name: str, left: Path, right: Path, left_label: str, right_label: str,
            left_duration: float, right_duration: float) -> Path:
    output = OUT / name
    max_duration = max(left_duration, right_duration)
    left_pad = max_duration - left_duration
    right_pad = max_duration - right_duration
    def chain(label: str, pad: float, color: str) -> str:
        return (
            "scale=960:960:flags=lanczos,pad=960:1080:0:120:color=0x0d121c,"
            f"tpad=stop_mode=clone:stop_duration={pad:.3f},"
            f"drawtext=fontfile={FONT}:text='{label}':fontcolor={color}:fontsize=35:"
            "x=(w-text_w)/2:y=42"
        )
    graph = f"[0:v]{chain(left_label,left_pad,'0x64d2ff')}[l];[1:v]{chain(right_label,right_pad,'0xffbe50')}[r];[l][r]hstack=inputs=2[v]"
    subprocess.run(
        [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-i", str(left), "-i", str(right),
         "-filter_complex", graph, "-map", "[v]", "-t", f"{max_duration:.3f}", "-r", "20",
         "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(output)], check=True,
    )
    return output


def main() -> None:
    if OUT.exists() or RESULT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT} or {RESULT}")
    OUT.mkdir(parents=True)
    base40 = ROOT / "outputs/presentation/p0_hd_openvla_base_init40_v2_seed20260936/clean_agentview_hd640.mp4"
    qlora40 = ROOT / "outputs/presentation/p0_hd_openvla_qlora_init40_v2_seed20260936/clean_agentview_hd640.mp4"
    base44 = ROOT / "outputs/presentation/p0_hd_openvla_base_init44_v2_seed20260940/clean_agentview_hd640.mp4"
    qlora44 = ROOT / "outputs/presentation/p0_hd_openvla_qlora_init44_v2_seed20260940/clean_agentview_hd640.mp4"
    sources = [base40, qlora40, base44, qlora44]
    assert all(p.is_file() for p in sources)
    typical = compose("base_vs_qlora_init40_typical_speedup_hd.mp4", base40, qlora40,
                      "FROZEN BASE  |  SUCCESS  |  221 steps",
                      "RANK-8 QLoRA  |  SUCCESS  |  70 steps", 11.05, 3.5)
    tail = compose("base_vs_qlora_init44_tail_failure_hd.mp4", base44, qlora44,
                   "FROZEN BASE  |  SUCCESS  |  68 steps",
                   "RANK-8 QLoRA  |  TIMEOUT  |  600 steps", 3.4, 30.0)
    outputs = []
    for p in (typical, tail):
        outputs.append({"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                        "sha256": digest(p), "probe": probe(p)})
    record = {
        "manifest": "p0_hd_comparisons_v1_20260828",
        "classification": "PASS",
        "generation_mode": "offline_composition_from_completed_hd_captures",
        "environment_steps": 0,
        "model_predictions": 0,
        "outputs": outputs,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sources],
        "interpretation": {
            "init40": "Both succeed; QLoRA uses 70 versus base 221 control steps, while per-step model inference is slower.",
            "init44": "Base succeeds in 68 steps; frozen QLoRA times out at 600 steps.",
        },
        "boundary": "Presentation-only composition. It cannot revise Phase 3A/3B selection, success estimates, or authorize tuning/final reruns.",
    }
    RESULT.write_text(yaml.safe_dump(record, sort_keys=False))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
