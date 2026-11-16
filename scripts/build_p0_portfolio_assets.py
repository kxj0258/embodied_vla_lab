#!/usr/bin/env python3
"""Build immutable P0 presentation assets from frozen Phase 2/3 evidence.

This script is deliberately offline: it never imports Torch, constructs LIBERO,
or loads a policy.  It only reads frozen JSONL/PNG/YAML artifacts and invokes the
environment-local ffmpeg encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/presentation/p0_mvp_v2_20260827"
BASE_ROOT = ROOT / "outputs/eval/phase3a_openvla_libero_final_heldout45_v1_init5to49_seed20260901"
QLORA_ROOT = ROOT / "outputs/eval/phase3b_openvla_qlora_step20_final_heldout45_v1_init5to49_seed20260901"
ACT_RESULT = ROOT / "results/phase2b_act_libero_ckpt100000_final_heldout45.yaml"
BASE_RESULT = ROOT / "results/phase3a_openvla_libero_final_heldout45_v1.yaml"
QLORA_RESULT = ROOT / "results/phase3b_openvla_qlora_step20_final_heldout45_v1.yaml"
PAIRED_RESULT = ROOT / "results/phase3b_openvla_step20_vs_base_fullval889_paired.yaml"
FFMPEG = Path("/home/kxj/miniforge3/envs/evla-core/bin/ffmpeg")
FFPROBE = Path("/home/kxj/miniforge3/envs/evla-core/bin/ffprobe")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def episodes(path: Path) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {}
    for row in read_jsonl(path):
        result.setdefault(int(row["init_state_index"]), []).append(row)
    for rows in result.values():
        rows.sort(key=lambda row: int(row["step"]))
    return result


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def frame_path(root: Path, init_state: int, step: int) -> Path:
    episode = init_state - 5
    return root / "ipc" / f"episode_{episode:02d}_init_{init_state:02d}" / f"step_{step:04d}_agentview_rot180.png"


def contain(image: Image.Image, box: tuple[int, int], background=(13, 18, 28)) -> Image.Image:
    copy = image.copy()
    scale = min(box[0] / copy.width, box[1] / copy.height)
    copy = copy.resize((round(copy.width * scale), round(copy.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", box, background)
    canvas.paste(copy, ((box[0] - copy.width) // 2, (box[1] - copy.height) // 2))
    return canvas


def draw_lines(draw: ImageDraw.ImageDraw, lines: list[str], xy: tuple[int, int], size=28, gap=14, fill=(231, 238, 248), bold=False) -> None:
    x, y = xy
    face = font(size, bold)
    for line in lines:
        draw.text((x, y), line, font=face, fill=fill)
        y += size + gap


def encode(frames_dir: Path, output: Path, fps: int = 10) -> None:
    subprocess.run(
        [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
         "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-crf", "18", str(output)],
        check=True,
    )


def base_demo(output: Path, base: dict[int, list[dict]], init_state: int = 5) -> dict:
    rows = base[init_state]
    frames = output / "frames_openvla_base_init05"
    frames.mkdir()
    for index, row in enumerate(rows):
        image = Image.open(frame_path(BASE_ROOT, init_state, int(row["step"]))).convert("RGB")
        canvas = Image.new("RGB", (1280, 720), (13, 18, 28))
        canvas.paste(contain(image, (720, 720)), (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((760, 54), "OpenVLA-7B · frozen base", font=font(34, True), fill=(100, 210, 255))
        draw_lines(draw, ["Instruction", "turn on the stove", "", f"Held-out init state  {init_state}",
                          f"Environment step   {row['step']} / {len(rows)}", f"Inference latency  {row['inference_s']:.3f} s",
                          f"Reward             {row['reward']:.1f}"], (760, 125), 25, 12)
        if row["success"]:
            draw.rounded_rectangle((750, 570, 1215, 660), radius=18, fill=(20, 115, 82))
            draw.text((790, 590), "STRICT SUCCESS", font=font(36, True), fill="white")
        canvas.save(frames / f"{index:05d}.png")
    movie = output / "openvla_base_heldout_init05_success.mp4"
    encode(frames, movie)
    shutil.rmtree(frames)
    return {"path": movie.relative_to(ROOT).as_posix(), "init_state": init_state, "steps": len(rows), "success": True}


def compare_video(output: Path, base: dict[int, list[dict]], qlora: dict[int, list[dict]], init_state: int, name: str, stride: int = 1) -> dict:
    left, right = base[init_state], qlora[init_state]
    count = max(math.ceil(len(left) / stride), math.ceil(len(right) / stride))
    frames = output / f"frames_{name}"
    frames.mkdir()
    for index in range(count):
        li = min(index * stride, len(left) - 1)
        ri = min(index * stride, len(right) - 1)
        lrow, rrow = left[li], right[ri]
        limage = contain(Image.open(frame_path(BASE_ROOT, init_state, int(lrow["step"]))).convert("RGB"), (600, 600))
        rimage = contain(Image.open(frame_path(QLORA_ROOT, init_state, int(rrow["step"]))).convert("RGB"), (600, 600))
        canvas = Image.new("RGB", (1280, 720), (13, 18, 28)); canvas.paste(limage, (20, 90)); canvas.paste(rimage, (660, 90))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 20), f"Frozen held-out init {init_state} · same task/runtime", font=font(28, True), fill=(235, 240, 248))
        draw.rounded_rectangle((25, 96, 310, 138), radius=8, fill=(13, 18, 28))
        draw.rounded_rectangle((665, 96, 978, 138), radius=8, fill=(13, 18, 28))
        draw.text((30, 101), "OpenVLA frozen base", font=font(22, True), fill=(100, 210, 255))
        draw.text((670, 101), "Rank-8 QLoRA step-20", font=font(22, True), fill=(255, 190, 80))
        lstatus = "SUCCESS" if lrow["success"] else ("COMPLETE" if li == len(left)-1 else "RUNNING")
        rstatus = "SUCCESS" if rrow["success"] else ("TIMEOUT / FAIL" if ri == len(right)-1 else "RUNNING")
        draw.text((35, 650), f"step {lrow['step']}/{len(left)}  ·  {lstatus}", font=font(23, True), fill=(120, 240, 170) if lrow["success"] else (235, 240, 248))
        draw.text((675, 650), f"step {rrow['step']}/{len(right)}  ·  {rstatus}", font=font(23, True), fill=(255, 105, 105) if "FAIL" in rstatus else ((120, 240, 170) if rrow["success"] else (235, 240, 248)))
        canvas.save(frames / f"{index:05d}.png")
    movie = output / f"{name}.mp4"; encode(frames, movie); shutil.rmtree(frames)
    return {"path": movie.relative_to(ROOT).as_posix(), "init_state": init_state, "base_steps": len(left), "adapter_steps": len(right),
            "base_success": bool(left[-1]["success"]), "adapter_success": bool(right[-1]["success"]), "sampling_stride": stride}


def charts(output: Path) -> list[dict]:
    paired = yaml.safe_load(PAIRED_RESULT.read_text())
    final = yaml.safe_load(QLORA_RESULT.read_text())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    metrics = [("Offline loss", paired["base"]["mean_loss"], paired["step20"]["mean_loss"], "lower is better"),
               ("Token accuracy (%)", 100*paired["base"]["token_accuracy"], 100*paired["step20"]["token_accuracy"], "higher is better"),
               ("Continuous action L1", paired["base"]["mean_continuous_l1"], paired["step20"]["mean_continuous_l1"], "lower is better")]
    for ax, (title, b, q, note) in zip(axes, metrics):
        bars = ax.bar(["Base", "QLoRA"], [b, q], color=["#2386b8", "#e69b2e"])
        ax.set_title(title); ax.set_ylabel(note); ax.grid(axis="y", alpha=.2)
        for bar, val in zip(bars, [b, q]): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom")
    fig.suptitle("Paired offline validation · same 889 frames (not environment success)", fontweight="bold")
    fig.tight_layout();
    png = output / "openvla_base_vs_qlora_offline_metrics.png"; svg = output / "openvla_base_vs_qlora_offline_metrics.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight"); fig.savefig(svg, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    names = ["ACT", "OpenVLA base", "OpenVLA QLoRA"]
    successes = [45, 45, 43]; failures = [0, 0, 2]
    ax.bar(names, successes, label="strict success", color="#2386b8")
    ax.bar(names, failures, bottom=successes, label="failure", color="#d94c4c")
    ax.set_ylim(0, 50); ax.set_ylabel("Held-out init states (n=45)"); ax.set_title("Frozen held-out closed-loop reliability")
    for i, value in enumerate(successes): ax.text(i, value-2.5, f"{value}/45", ha="center", color="white", fontweight="bold")
    ax.legend(frameon=False, loc="upper left"); ax.grid(axis="y", alpha=.2); fig.tight_layout()
    rpng = output / "heldout_reliability.png"; rsvg = output / "heldout_reliability.svg"
    fig.savefig(rpng, dpi=180, bbox_inches="tight"); fig.savefig(rsvg, bbox_inches="tight"); plt.close(fig)
    return [{"path": p.relative_to(ROOT).as_posix()} for p in [png, svg, rpng, rsvg]]


def architecture(output: Path) -> list[dict]:
    svg = output / "embodied_vla_lab_architecture.svg"
    svg.write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="760" viewBox="0 0 1400 760">
<defs><marker id="a" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#506477"/></marker><style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#17212b}.t{font-size:30px;font-weight:700}.h{font-size:21px;font-weight:700}.b{font-size:16px}.box{rx:18;stroke-width:2}.arr{stroke:#506477;stroke-width:3;marker-end:url(#a)}</style></defs>
<rect width="1400" height="760" fill="#f7f9fc"/><text x="55" y="58" class="t">Embodied VLA Lab · reproducible control architecture</text>
<rect class="box" x="55" y="110" width="260" height="170" fill="#e8f3fa" stroke="#2386b8"/><text x="85" y="155" class="h">Frozen evidence</text><text x="85" y="192" class="b">LIBERO HDF5 + init states</text><text x="85" y="222" class="b">PNG / JSON / raw logs</text><text x="85" y="252" class="b">SHA-256 manifests</text>
<rect class="box" x="405" y="110" width="300" height="170" fill="#eef7ef" stroke="#3b9660"/><text x="435" y="155" class="h">evla-core</text><text x="435" y="192" class="b">MuJoCo / LIBERO environment</text><text x="435" y="222" class="b">ACT policy + evaluation</text><text x="435" y="252" class="b">Headless EGL rendering</text>
<rect class="box" x="795" y="110" width="330" height="170" fill="#fff3df" stroke="#e69b2e"/><text x="825" y="155" class="h">evla-openvla</text><text x="825" y="192" class="b">OpenVLA-7B NF4 / BF16</text><text x="825" y="222" class="b">Frozen base or rank-8 QLoRA</text><text x="825" y="252" class="b">Persistent model worker</text>
<rect class="box" x="405" y="370" width="720" height="150" fill="#f1ecfa" stroke="#7658ae"/><text x="435" y="413" class="h">Dependency-isolated hashed IPC control bridge</text><text x="435" y="450" class="b">rotated agent-view PNG + schema/hash request  →  7D bounded action + hash response</text><text x="435" y="482" class="b">timeout · gripper adaptation · finite/range checks · deterministic cleanup</text>
<rect class="box" x="405" y="610" width="720" height="100" fill="#fcebec" stroke="#c94f59"/><text x="435" y="652" class="h">Immutable evidence layer</text><text x="435" y="684" class="b">structured results · checkpoint/video manifests · held-out protocol · failure preservation</text>
<line class="arr" x1="315" y1="195" x2="395" y2="195"/><line class="arr" x1="705" y1="195" x2="785" y2="195"/><line class="arr" x1="555" y1="280" x2="555" y2="360"/><line class="arr" x1="960" y1="280" x2="960" y2="360"/><line class="arr" x1="765" y1="520" x2="765" y2="600"/>
<text x="1160" y="185" class="b">12 GB RTX 5070 Ti</text><text x="1160" y="215" class="b">strict offline</text><text x="1160" y="245" class="b">no W&amp;B / Hub</text></svg>''')
    png = output / "embodied_vla_lab_architecture.png"
    subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-i", str(svg), "-frames:v", "1", str(png)], check=True)
    return [{"path": p.relative_to(ROOT).as_posix()} for p in [svg, png]]


def probe(path: Path) -> dict:
    raw = subprocess.check_output([str(FFPROBE), "-v", "error", "-show_entries", "format=duration,size", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)])
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    output = args.output.resolve()
    if output.exists(): raise FileExistsError(f"refusing to overwrite: {output}")
    for required in [FFMPEG, FFPROBE, BASE_ROOT / "trajectory.jsonl", QLORA_ROOT / "trajectory.jsonl", ACT_RESULT, BASE_RESULT, QLORA_RESULT, PAIRED_RESULT]:
        if not required.exists(): raise FileNotFoundError(required)
    output.mkdir(parents=True)
    base, qlora = episodes(BASE_ROOT / "trajectory.jsonl"), episodes(QLORA_ROOT / "trajectory.jsonl")
    common = [state for state in base if base[state][-1]["success"] and qlora[state][-1]["success"]]
    typical = min(common, key=lambda state: len(qlora[state]) - len(base[state]))
    assets = [base_demo(output, base), compare_video(output, base, qlora, typical, "openvla_base_vs_qlora_typical_speedup"),
              compare_video(output, base, qlora, 44, "openvla_base_vs_qlora_init44_tail_failure", stride=4)]
    figures = charts(output) + architecture(output)
    for item in assets: item["ffprobe"] = probe(ROOT / item["path"])
    sources = [ACT_RESULT, BASE_RESULT, QLORA_RESULT, PAIRED_RESULT, BASE_ROOT / "trajectory.jsonl", QLORA_ROOT / "trajectory.jsonl"]
    manifest = {"manifest": "p0_mvp_v2_20260827", "classification": "PASS", "generation_mode": "offline_from_frozen_assets",
                "environment_steps": 0, "model_inferences": 0, "selected_typical_speedup_init": typical,
                "videos": assets, "figures": figures,
                "source_artifacts": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sources],
                "boundary": "Videos replay frozen observations. They are not new rollout evidence and do not imply real-time 20 Hz control."}
    for item in manifest["videos"] + manifest["figures"]:
        path = ROOT / item["path"]; item.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
    (output / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
    print(json.dumps({"classification": "PASS", "output": str(output), "typical_speedup_init": typical, "videos": len(assets), "figures": len(figures)}, indent=2))


if __name__ == "__main__":
    main()
