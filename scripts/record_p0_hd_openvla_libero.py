#!/usr/bin/env python3
"""Generic 640px LIBERO presentation capture with frozen 128px OpenVLA IPC input."""
from __future__ import annotations
import json, os, random, select, shutil, subprocess, sys, time
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1"); os.environ.setdefault("WANDB_MODE","disabled"); os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("MUJOCO_EGL_DEVICE_ID","0")
import numpy as np, psutil, yaml
from PIL import Image, ImageDraw, ImageFont
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
from phase3a_openvla_bridge_protocol import SCHEMA_VERSION,load_json,sha256,validate_response,write_json_atomic
FFMPEG=Path("/home/kxj/miniforge3/envs/evla-core/bin/ffmpeg")
def face(n,b=False): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if b else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",n)
def event(worker,timeout):
    ready,_,_=select.select([worker.stdout],[],[],timeout)
    if not ready: raise TimeoutError("worker timeout")
    line=worker.stdout.readline()
    if not line: raise RuntimeError(f"worker closed: {worker.poll()}")
    value=json.loads(line); print(json.dumps({"worker_event":value}),flush=True); return value
def scalar(x):
    a=np.asarray(x); assert a.size==1; return a.reshape(-1)[0].item()
def compose(raw_dir:Path,rows:list[dict],out:Path,label:str,instruction:str,init_state:int):
    temp=out/"presentation_frames"; temp.mkdir()
    for i,row in enumerate(rows):
        raw=Image.open(raw_dir/f"{i:05d}.png").convert("RGB"); canvas=Image.new("RGB",(1920,1080),(13,18,28)); canvas.paste(raw.resize((1080,1080),Image.Resampling.LANCZOS),(0,0)); d=ImageDraw.Draw(canvas)
        color=(100,210,255) if "base" in label.lower() else (255,190,80); d.text((1140,70),label,font=face(40,True),fill=color); d.text((1140,160),f"Instruction\n{instruction}",font=face(31),fill=(235,240,248),spacing=12)
        d.text((1140,360),f"640×640 source capture\n128×128 frozen model IPC\nHeld-out init state  {init_state}\nEnvironment step   {row['step']}\nInference          {row['inference_s']:.3f} s\nReward sum         {row['reward_sum']:.1f}",font=face(29),fill=(235,240,248),spacing=16)
        if row["success"]: d.rounded_rectangle((1125,870,1845,1010),radius=24,fill=(20,115,82)); d.text((1190,905),"STRICT SUCCESS",font=face(48,True),fill="white")
        elif row["step"]==rows[-1]["step"]: d.rounded_rectangle((1125,870,1845,1010),radius=24,fill=(150,48,52)); d.text((1210,905),"TIMEOUT / FAIL",font=face(46,True),fill="white")
        canvas.save(temp/f"{i:05d}.png",compress_level=3)
    movie=out/"presentation_hd640.mp4"; clean=out/"clean_agentview_hd640.mp4"
    common=[str(FFMPEG),"-hide_banner","-loglevel","error","-framerate","20","-i"]
    subprocess.run(common+[str(raw_dir/"%05d.png"),"-vf","scale=1080:1080:flags=lanczos","-c:v","libx264","-preset","slow","-crf","16","-pix_fmt","yuv420p","-movflags","+faststart",str(clean)],check=True)
    subprocess.run(common+[str(temp/"%05d.png"),"-c:v","libx264","-preset","slow","-crf","16","-pix_fmt","yuv420p","-movflags","+faststart",str(movie)],check=True)
    shutil.rmtree(raw_dir); shutil.rmtree(temp); return clean,movie
def main():
    assert os.environ.get("LIBERO_CONFIG_PATH")==str(ROOT/"configs/libero/phase2a_runtime") and not (Path.home()/".libero/config.yaml").exists()
    config_path=Path(os.environ["EVLA_P0_HD_CONFIG"]); c=yaml.safe_load(config_path.read_text()); assert c["protocol"]["schema_version"]==SCHEMA_VERSION
    out=ROOT/c["output_dir"]; result_path=ROOT/c["structured_result"]
    for p in [out,result_path]:
        if p.exists(): raise FileExistsError(f"refusing overwrite: {p}")
    out.mkdir(parents=True); ipc=out/"ipc"; ipc.mkdir(); raw=out/"source_frames_640_lossless"; raw.mkdir(); (out/"resolved_config.yaml").write_text(yaml.safe_dump(c,sort_keys=False))
    seed=int(c["protocol"]["evaluator_seed"]); init_state=int(c["protocol"]["init_state_index"]); random.seed(seed); np.random.seed(seed)
    suite=benchmark.get_benchmark_dict()[c["task"]["suite"]](task_order_index=0); task_id=int(c["task"]["task_id"]); assert suite.get_task(task_id).name==c["task"]["name"] and suite.get_task(task_id).language==c["task"]["instruction"]
    env=LiberoEnv(task_suite=suite,task_id=task_id,task_suite_name=c["task"]["suite"],episode_length=600,camera_name=["agentview_image","robot0_eye_in_hand_image"],camera_name_mapping={"agentview_image":"agentview","robot0_eye_in_hand_image":"eye_in_hand"},obs_type="pixels_agent_pos",observation_width=128,observation_height=128,init_states=True,episode_index=init_state,n_envs=1,num_steps_wait=5,control_freq=20,control_mode="relative",hard_reset=True)
    worker=None; rows=[]; started=time.perf_counter(); process=psutil.Process(); latest=None
    try:
        obs,_=env.reset(seed=seed); worker=subprocess.Popen([c["model"]["python"],str(ROOT/c["model"]["worker"])],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1,env={**os.environ,"CUDA_VISIBLE_DEVICES":"0"}); ready=event(worker,180); assert ready["classification"]=="PASS"
        reward_sum=0.; success=False; instruction=c["task"]["instruction"]; prompt=f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        trajectory=out/"trajectory.jsonl"
        with trajectory.open("x") as tf:
            for step in range(1,601):
                assert env._env is not None
                high=np.asarray(env._env.sim.render(width=640,height=640,camera_name="agentview")); assert high.shape==(640,640,3); display=high[::-1,::-1].copy(); Image.fromarray(display).save(raw/f"{step-1:05d}.png",compress_level=3)
                native=np.asarray(obs["pixels"]["agentview"]); assert native.shape==(128,128,3); model=native[::-1,::-1].copy(); image_path=ipc/f"step_{step:04d}_model128.png"; Image.fromarray(model).save(image_path)
                req_path=ipc/f"step_{step:04d}_request.json"; resp_path=ipc/f"step_{step:04d}_response.json"; req={"schema_version":SCHEMA_VERSION,"experiment_id":c["experiment_id"],"request_id":f"{c['experiment_id']}:step:{step:04d}","checkpoint_revision":c["model"]["revision"],"unnorm_key":"libero_goal","instruction":instruction,"prompt":prompt,"image_path":image_path.name,"image_sha256":sha256(image_path),"image_shape":[128,128,3],"image_dtype":"uint8","preprocess":c["protocol"]["model_image_preprocess"],"environment_steps_requested":1}; write_json_atomic(req_path,req)
                worker.stdin.write(json.dumps({"command":"predict","request":str(req_path),"response":str(resp_path)})+"\n"); worker.stdin.flush(); ev=event(worker,180); assert ev["request_id"]==req["request_id"]
                resp=load_json(resp_path); latest=resp; action=validate_response(resp,req); obs,reward,terminated,truncated,info=env.step(action); reward_sum+=float(scalar(reward)); success=success or bool(info.get("is_success",False)); row={"step":step,"reward":float(scalar(reward)),"reward_sum":reward_sum,"success":success,"terminated":bool(scalar(terminated)),"truncated":bool(scalar(truncated)),"inference_s":float(resp["inference_s"]),"source_image_sha256":sha256(raw/f"{step-1:05d}.png"),"model_image_sha256":sha256(image_path)}; rows.append(row); tf.write(json.dumps(row)+"\n"); tf.flush()
                if row["terminated"] or row["truncated"]: break
        worker.stdin.write(json.dumps({"command":"shutdown"})+"\n"); worker.stdin.flush(); shutdown=event(worker,30); assert shutdown["predictions"]==len(rows); worker.wait(30); assert worker.returncode==0
    finally:
        if worker is not None and worker.poll() is None: worker.terminate(); worker.wait(10)
        env.close()
    clean,movie=compose(raw,rows,out,c["presentation_label"],c["task"]["instruction"],init_state); trajectory=out/"trajectory.jsonl"; inf=[x["inference_s"] for x in rows]
    result={"classification":"PASS","experiment":c["experiment_id"],"execution_pass":True,"policy_success":rows[-1]["success"],"steps":len(rows),"sum_reward":rows[-1]["reward_sum"],"model_predictions":len(rows),"model_load_s":ready["model_load_s"],"total_inference_s":sum(inf),"median_inference_s":float(np.median(inf)),"p95_inference_s":float(np.percentile(inf,95)),"elapsed_s":time.perf_counter()-started,"render_resolution":[640,640],"model_ipc_resolution":[128,128],"peak_vram_allocated_bytes":latest["peak_vram_allocated_bytes"],"peak_vram_reserved_bytes":latest["peak_vram_reserved_bytes"],"worker_rss_kib":latest["worker_rss_kib_at_result"],"coordinator_rss_kib":process.memory_info().rss//1024,"artifacts":{"presentation_video":{"path":movie.relative_to(ROOT).as_posix(),"bytes":movie.stat().st_size,"sha256":sha256(movie)},"clean_video":{"path":clean.relative_to(ROOT).as_posix(),"bytes":clean.stat().st_size,"sha256":sha256(clean)},"trajectory":{"path":trajectory.relative_to(ROOT).as_posix(),"sha256":sha256(trajectory)}},"boundary":c["boundary"]}; result_path.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
