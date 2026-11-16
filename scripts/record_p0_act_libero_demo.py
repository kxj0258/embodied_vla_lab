#!/usr/bin/env python3
"""Capture one immutable headless ACT-on-LIBERO presentation episode."""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/evla_p0_act_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/evla_p0_act_matplotlib")

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv

from evaluate_phase2b_act_libero import ROOT, policy_batch, policy_config, sha256
from lerobot.policies.act.modeling_act import ACTPolicy


CONFIG = ROOT / "configs/eval/act_libero_p0_demo_init05_v1.yaml"
FFMPEG = Path("/home/kxj/miniforge3/envs/evla-core/bin/ffmpeg")


def face(size: int, bold: bool = False):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size)


def main() -> None:
    assert os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("WANDB_MODE") == "disabled"
    assert os.environ.get("LIBERO_CONFIG_PATH") and not (Path.home() / ".libero/config.yaml").exists()
    config = yaml.safe_load(CONFIG.read_text()); output = ROOT / config["output_dir"]
    result_path = ROOT / config["structured_result"]
    # The host command creates the raw log through tee after checking it itself.
    for target in [output, result_path]:
        if target.exists(): raise FileExistsError(f"refusing to overwrite: {target}")
    output.mkdir(parents=True); frames = output / "frames"; frames.mkdir()
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    checkpoint_path = ROOT / config["checkpoint"]
    assert sha256(checkpoint_path) == config["checkpoint_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint["stats"].items()}
    seed = int(config["protocol"]["evaluator_seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); device = torch.device("cuda:0")
    policy = ACTPolicy(policy_config()).to(device).eval(); policy.load_state_dict(checkpoint["model"], strict=True)
    suite = benchmark.get_benchmark_dict()[config["task"]["suite"]](task_order_index=config["task"]["task_order_index"])
    env = LiberoEnv(task_suite=suite, task_id=config["task"]["task_id"], task_suite_name=config["task"]["suite"],
        episode_length=config["protocol"]["episode_horizon"], camera_name=["agentview_image", "robot0_eye_in_hand_image"],
        camera_name_mapping={"agentview_image":"agentview", "robot0_eye_in_hand_image":"eye_in_hand"}, obs_type="pixels_agent_pos",
        observation_width=128, observation_height=128, init_states=True, episode_index=5, n_envs=1,
        num_steps_wait=config["protocol"]["physics_settle_steps"], control_freq=config["protocol"]["control_frequency_hz"], control_mode="relative", hard_reset=True)
    trajectory=[]; reward_sum=0.0; inference_s=0.0; started=time.perf_counter(); success=False
    try:
        observation,_=env.reset(seed=seed); policy.reset()
        for step in range(1, config["protocol"]["episode_horizon"]+1):
            image=np.rot90(np.asarray(observation["pixels"]["agentview"]),2).copy()
            batch=policy_batch(observation,stats,device); tick=time.perf_counter(); normalized=policy.select_action(batch); torch.cuda.synchronize(); inference_s += time.perf_counter()-tick
            action=np.clip(normalized[0].float().cpu().numpy()*stats["as"]+stats["am"],-1,1).astype(np.float32)
            observation,reward,terminated,truncated,info=env.step(action); reward_sum += float(reward); success = success or bool(info.get("is_success",False))
            canvas=Image.new("RGB",(1280,720),(13,18,28)); visual=Image.fromarray(image).resize((720,720),Image.Resampling.LANCZOS); canvas.paste(visual,(0,0)); draw=ImageDraw.Draw(canvas)
            draw.text((760,55),"ACT · deterministic · step 100k",font=face(31,True),fill=(100,210,255)); draw.text((760,125),"Instruction\nturn on the stove",font=face(26),fill=(235,240,248),spacing=12)
            draw.text((760,245),f"Held-out init state  5\nEnvironment step   {step}\nReward sum         {reward_sum:.1f}\nPolicy window      5 actions",font=face(24),fill=(235,240,248),spacing=15)
            if success: draw.rounded_rectangle((750,570,1215,660),radius=18,fill=(20,115,82)); draw.text((790,590),"STRICT SUCCESS",font=face(36,True),fill="white")
            canvas.save(frames/f"{step-1:05d}.png")
            trajectory.append({"step":step,"reward":float(reward),"success":success,"action":action.tolist()})
            if terminated or truncated: break
    finally: env.close()
    movie=output/"act_libero_ckpt100000_heldout_init05_success.mp4"
    subprocess.run([str(FFMPEG),"-hide_banner","-loglevel","error","-framerate",str(config["protocol"]["capture_fps"]),"-i",str(frames/"%05d.png"),"-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart","-crf","18",str(movie)],check=True)
    import shutil; shutil.rmtree(frames)
    (output/"trajectory.jsonl").write_text("".join(json.dumps(row)+"\n" for row in trajectory))
    result={"experiment":config["experiment_id"],"classification":"PASS","execution_pass":True,"policy_success":success,"steps":len(trajectory),"sum_reward":reward_sum,
        "inference_s":inference_s,"elapsed_s":time.perf_counter()-started,"peak_vram_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_vram_reserved_bytes":torch.cuda.max_memory_reserved(),
        "artifacts":{"video":{"path":movie.relative_to(ROOT).as_posix(),"bytes":movie.stat().st_size,"sha256":sha256(movie)},"trajectory":{"path":(output/"trajectory.jsonl").relative_to(ROOT).as_posix(),"sha256":sha256(output/"trajectory.jsonl")}},"boundary":config["boundary"]}
    result_path.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
