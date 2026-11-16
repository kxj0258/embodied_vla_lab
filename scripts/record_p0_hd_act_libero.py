#!/usr/bin/env python3
"""One 640px LIBERO capture while preserving ACT's frozen 128px input contract."""
from __future__ import annotations
import json, os, random, shutil, subprocess, time
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("WANDB_MODE","disabled"); os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("MUJOCO_EGL_DEVICE_ID","0")
import numpy as np, torch, yaml
from PIL import Image, ImageDraw, ImageFont
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
from lerobot.policies.act.modeling_act import ACTPolicy
from evaluate_phase2b_act_libero import ROOT, policy_config, sha256

CONFIG=ROOT/"configs/eval/p0_hd_act_libero_init05_v1.yaml"; FFMPEG=Path("/home/kxj/miniforge3/envs/evla-core/bin/ffmpeg")
def face(n,b=False): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if b else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",n)
def batch_hd(obs,stats,device):
    result={}
    for source in ["agentview","eye_in_hand"]:
        raw=np.asarray(obs["pixels"][source]); assert raw.shape==(640,640,3) and raw.dtype==np.uint8
        rotated=raw[::-1,::-1].copy(); small=np.asarray(Image.fromarray(rotated).resize((128,128),Image.Resampling.BILINEAR))
        result[f"observation.images.{source}"]=torch.from_numpy(small.copy()).permute(2,0,1).float().div_(255).unsqueeze(0).to(device)
    robot=obs["robot_state"]; state=np.concatenate([robot["joints"]["pos"],robot["gripper"]["qpos"]]).astype(np.float32)
    result["observation.state"]=torch.from_numpy((state-stats["sm"])/stats["ss"]).unsqueeze(0).to(device); return result
def main():
    assert os.environ.get("LIBERO_CONFIG_PATH")==str(ROOT/"configs/libero/phase2a_runtime") and not (Path.home()/".libero/config.yaml").exists()
    c=yaml.safe_load(CONFIG.read_text()); out=ROOT/c["output_dir"]; result_path=ROOT/c["structured_result"]
    for p in [out,result_path]:
        if p.exists(): raise FileExistsError(f"refusing to overwrite: {p}")
    out.mkdir(parents=True); frames=out/"frames_640_lossless"; frames.mkdir(); (out/"resolved_config.yaml").write_text(yaml.safe_dump(c,sort_keys=False))
    cp=ROOT/c["checkpoint"]; actual=sha256(cp); assert actual==c["checkpoint_sha256"]
    ckpt=torch.load(cp,map_location="cpu",weights_only=False); stats={k:np.asarray(v,dtype=np.float32) for k,v in ckpt["stats"].items()}
    seed=c["protocol"]["evaluator_seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device=torch.device("cuda:0"); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); policy=ACTPolicy(policy_config()).to(device).eval(); policy.load_state_dict(ckpt["model"],strict=True)
    suite=benchmark.get_benchmark_dict()[c["task"]["suite"]](task_order_index=0); env=LiberoEnv(task_suite=suite,task_id=7,task_suite_name=c["task"]["suite"],episode_length=600,camera_name=["agentview_image","robot0_eye_in_hand_image"],camera_name_mapping={"agentview_image":"agentview","robot0_eye_in_hand_image":"eye_in_hand"},obs_type="pixels_agent_pos",observation_width=640,observation_height=640,init_states=True,episode_index=5,n_envs=1,num_steps_wait=5,control_freq=20,control_mode="relative",hard_reset=True)
    rows=[]; reward_sum=0.; success=False; infer=0.; started=time.perf_counter()
    try:
        obs,_=env.reset(seed=seed); policy.reset()
        for step in range(1,601):
            display=np.asarray(obs["pixels"]["agentview"])[::-1,::-1].copy(); tick=time.perf_counter(); norm=policy.select_action(batch_hd(obs,stats,device)); torch.cuda.synchronize(); infer+=time.perf_counter()-tick
            action=np.clip(norm[0].float().cpu().numpy()*stats["as"]+stats["am"],-1,1).astype(np.float32); obs,reward,terminated,truncated,info=env.step(action); reward_sum+=float(reward); success=success or bool(info.get("is_success",False))
            canvas=Image.new("RGB",(1920,1080),(13,18,28)); canvas.paste(Image.fromarray(display).resize((1080,1080),Image.Resampling.LANCZOS),(0,0)); d=ImageDraw.Draw(canvas)
            d.text((1140,70),"ACT · deterministic · step-100k",font=face(40,True),fill=(100,210,255)); d.text((1140,160),"Instruction\nturn on the stove",font=face(34),fill=(235,240,248),spacing=15); d.text((1140,360),f"640×640 source capture\n128×128 frozen policy input\nHeld-out init state  5\nEnvironment step   {step}\nReward sum         {reward_sum:.1f}",font=face(30),fill=(235,240,248),spacing=17)
            if success: d.rounded_rectangle((1125,870,1845,1010),radius=24,fill=(20,115,82)); d.text((1190,905),"STRICT SUCCESS",font=face(48,True),fill="white")
            canvas.save(frames/f"{step-1:05d}.png",compress_level=3); rows.append({"step":step,"reward":float(reward),"success":success,"action":action.tolist()})
            if terminated or truncated: break
    finally: env.close()
    movie=out/"act_libero_hd640_init05_success.mp4"; subprocess.run([str(FFMPEG),"-hide_banner","-loglevel","error","-framerate","20","-i",str(frames/"%05d.png"),"-c:v","libx264","-preset","slow","-crf","16","-pix_fmt","yuv420p","-movflags","+faststart",str(movie)],check=True); shutil.rmtree(frames)
    traj=out/"trajectory.jsonl"; traj.write_text("".join(json.dumps(x)+"\n" for x in rows)); result={"classification":"PASS","experiment":c["experiment_id"],"execution_pass":True,"policy_success":success,"steps":len(rows),"sum_reward":reward_sum,"policy_forwards":(len(rows)+4)//5,"total_inference_s":infer,"elapsed_s":time.perf_counter()-started,"render_resolution":[640,640],"policy_resolution":[128,128],"peak_vram_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_vram_reserved_bytes":torch.cuda.max_memory_reserved(),"checkpoint_sha256":actual,"artifacts":{"video":{"path":movie.relative_to(ROOT).as_posix(),"bytes":movie.stat().st_size,"sha256":sha256(movie)},"trajectory":{"path":traj.relative_to(ROOT).as_posix(),"sha256":sha256(traj)}},"boundary":c["boundary"]}; result_path.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
