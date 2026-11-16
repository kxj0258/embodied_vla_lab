#!/usr/bin/env python3
"""Fix presentation layout of the immutable ACT capture without rerunning policy/environment."""
from __future__ import annotations
import hashlib, json, shutil, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"outputs/presentation/p0_act_libero_ckpt100000_demo_init05_v1_seed20260901/act_libero_ckpt100000_heldout_init05_success.mp4"
OUTPUT=ROOT/"outputs/presentation/p0_act_libero_demo_presentation_v2_20260827"
FFMPEG=Path("/home/kxj/miniforge3/envs/evla-core/bin/ffmpeg")
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def main():
    if OUTPUT.exists(): raise FileExistsError(f"refusing to overwrite: {OUTPUT}")
    OUTPUT.mkdir(parents=True); frames=OUTPUT/"frames"; frames.mkdir()
    subprocess.run([str(FFMPEG),"-hide_banner","-loglevel","error","-i",str(SOURCE),str(frames/"%05d.png")],check=True)
    title_font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",28)
    for path in sorted(frames.glob("*.png")):
        image=Image.open(path).convert("RGB"); draw=ImageDraw.Draw(image)
        draw.rectangle((720,35,1279,105),fill=(13,18,28))
        draw.text((760,55),"ACT · deterministic · step-100k",font=title_font,fill=(100,210,255))
        image.save(path)
    movie=OUTPUT/"act_libero_ckpt100000_heldout_init05_success_presentation_v2.mp4"
    subprocess.run([str(FFMPEG),"-hide_banner","-loglevel","error","-framerate","10","-i",str(frames/"%05d.png"),"-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart","-crf","18",str(movie)],check=True)
    shutil.rmtree(frames)
    result={"classification":"PASS","generation_mode":"offline_recomposition_from_immutable_capture","environment_steps":0,"model_inferences":0,
      "source":{"path":SOURCE.relative_to(ROOT).as_posix(),"bytes":SOURCE.stat().st_size,"sha256":sha(SOURCE)},
      "video":{"path":movie.relative_to(ROOT).as_posix(),"bytes":movie.stat().st_size,"sha256":sha(movie)},
      "boundary":"Layout-only recomposition. Policy trajectory and frozen Phase 2 evidence are unchanged."}
    (OUTPUT/"manifest.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
