#!/usr/bin/env python3
"""Frozen OpenVLA-base configurable multi-task LIBERO-Goal breadth gate."""
from __future__ import annotations
import json, os, random, subprocess, sys, time
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1"); os.environ.setdefault("WANDB_MODE","disabled"); os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("MUJOCO_EGL_DEVICE_ID","0")
import numpy as np, psutil, yaml
from PIL import Image
from lerobot.envs.libero import LiberoEnv
from libero.libero import benchmark
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
from evaluate_phase3a_openvla_libero_single_episode import read_worker_event,scalar
from phase3a_openvla_bridge_protocol import SCHEMA_VERSION,load_json,sha256,validate_response,write_json_atomic

def main():
    assert os.environ.get("LIBERO_CONFIG_PATH")==str(ROOT/"configs/libero/phase2a_runtime")
    if (Path.home()/".libero/config.yaml").exists(): raise RuntimeError("refusing user-home LIBERO config")
    cp=Path(os.environ["EVLA_P0_BREADTH_CONFIG"]); c=yaml.safe_load(cp.read_text()); p=c["protocol"]
    assert p["schema_version"]==SCHEMA_VERSION and p["init_state_indices"]==[0,1,2] and p["episodes_per_task"]==3
    assert p["total_episodes"]==len(c["tasks"])*p["episodes_per_task"]
    out=ROOT/c["output_dir"]; result_path=ROOT/c["structured_result"]
    for target in (out,result_path):
        if target.exists(): raise FileExistsError(f"refusing overwrite: {target}")
    out.mkdir(parents=True); ipc=out/"ipc"; ipc.mkdir(); (out/"resolved_config.yaml").write_text(yaml.safe_dump(c,sort_keys=False))
    seed0=int(p["evaluator_seed_start"]); random.seed(seed0); np.random.seed(seed0)
    suite=benchmark.get_benchmark_dict()[c["suite"]["name"]](task_order_index=int(c["suite"]["task_order_index"])); expected=list(range(suite.n_tasks))
    assert [int(x["task_id"]) for x in c["tasks"]]==expected
    for task in c["tasks"]:
        meta=suite.get_task(int(task["task_id"])); assert meta.name==task["name"] and meta.language==task["instruction"]
    worker=None; env=None; latest=None; episodes=[]; all_inf=[]; predictions=0; started=time.perf_counter(); proc=psutil.Process(); trajectory=out/"trajectory.jsonl"
    try:
        worker=subprocess.Popen([c["model"]["openvla_python"],str(ROOT/c["model"]["worker"])],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1,env={**os.environ,"CUDA_VISIBLE_DEVICES":"0"})
        ready=read_worker_event(worker,int(p["worker_start_timeout_s"])); assert ready["classification"]=="PASS" and ready["checkpoint_revision"]==c["model"]["revision"]
        with trajectory.open("x") as tf:
            global_episode=0
            for task in c["tasks"]:
                task_id=int(task["task_id"]); env=LiberoEnv(task_suite=suite,task_id=task_id,task_suite_name=c["suite"]["name"],episode_length=int(p["episode_horizon"]),camera_name=["agentview_image","robot0_eye_in_hand_image"],camera_name_mapping={"agentview_image":"agentview","robot0_eye_in_hand_image":"eye_in_hand"},obs_type="pixels_agent_pos",observation_width=128,observation_height=128,init_states=True,episode_index=0,n_envs=1,num_steps_wait=int(p["physics_settle_steps"]),control_freq=int(p["control_frequency_hz"]),control_mode="relative",hard_reset=True)
                try:
                    for local_episode,init_index in enumerate(p["init_state_indices"]):
                        seed=seed0+global_episode; obs,_=env.reset(seed=seed); epdir=ipc/f"task_{task_id:02d}_{task['name']}/init_{init_index:02d}"; epdir.mkdir(parents=True)
                        reward_sum=0.; max_reward=float("-inf"); success=False; ep_inf=0.; ep_start=time.perf_counter(); term=trunc=False; steps=0
                        prompt=f"In: What action should the robot take to {task['instruction'].lower()}?\nOut:"
                        for step in range(1,int(p["episode_horizon"])+1):
                            image=np.asarray(obs["pixels"]["agentview"]); assert image.shape==(128,128,3) and image.dtype==np.uint8
                            model_image=image[::-1,::-1].copy(); image_path=epdir/f"step_{step:04d}_model128.png"; Image.fromarray(model_image).save(image_path)
                            req_path=epdir/f"step_{step:04d}_request.json"; resp_path=epdir/f"step_{step:04d}_response.json"; request_id=f"{c['experiment_id']}:task:{task_id:02d}:init:{init_index:02d}:step:{step:04d}"
                            req={"schema_version":SCHEMA_VERSION,"experiment_id":c["experiment_id"],"request_id":request_id,"checkpoint_revision":c["model"]["revision"],"unnorm_key":c["model"]["unnorm_key"],"instruction":task["instruction"],"prompt":prompt,"image_path":image_path.name,"image_sha256":sha256(image_path),"image_shape":[128,128,3],"image_dtype":"uint8","preprocess":p["model_image_preprocess"],"environment_steps_requested":1}
                            write_json_atomic(req_path,req); worker.stdin.write(json.dumps({"command":"predict","request":str(req_path),"response":str(resp_path)})+"\n"); worker.stdin.flush(); event=read_worker_event(worker,int(p["per_action_timeout_s"])); assert event["request_id"]==request_id
                            resp=load_json(resp_path); latest=resp; action=validate_response(resp,req); obs,reward,terminated,truncated,info=env.step(action); predictions+=1; steps=step
                            rv=float(scalar(reward)); reward_sum+=rv; max_reward=max(max_reward,rv); success=success or bool(info.get("is_success",False)); term=bool(scalar(terminated)); trunc=bool(scalar(truncated)); inf=float(resp["inference_s"]); ep_inf+=inf; all_inf.append(inf)
                            tf.write(json.dumps({"global_episode":global_episode,"task_id":task_id,"task_name":task["name"],"instruction":task["instruction"],"init_state_index":init_index,"seed":seed,"step":step,"reward":rv,"reward_sum":reward_sum,"success":success,"terminated":term,"truncated":trunc,"inference_s":inf,"action":resp["libero_adapted_action"],"image_sha256":req["image_sha256"],"response_sha256":sha256(resp_path)})+"\n"); tf.flush()
                            if term or trunc: break
                        summary={"global_episode":global_episode,"task_id":task_id,"task_name":task["name"],"instruction":task["instruction"],"init_state_index":init_index,"seed":seed,"success":success,"steps":steps,"sum_reward":reward_sum,"max_reward":max_reward,"terminated":term,"truncated":trunc,"inference_s":ep_inf,"episode_wall_s":time.perf_counter()-ep_start}; episodes.append(summary); print(json.dumps({"episode_summary":summary}),flush=True); global_episode+=1
                finally:
                    env.close(); env=None
        worker.stdin.write(json.dumps({"command":"shutdown"})+"\n"); worker.stdin.flush(); shutdown=read_worker_event(worker,30); assert shutdown["classification"]=="PASS" and shutdown["predictions"]==predictions; worker.wait(30); assert worker.returncode==0 and latest is not None
        task_summaries=[]
        for task in c["tasks"]:
            rows=[x for x in episodes if x["task_id"]==int(task["task_id"])]; task_summaries.append({"task_id":task["task_id"],"task_name":task["name"],"instruction":task["instruction"],"success_count":sum(int(x["success"]) for x in rows),"success_total":len(rows),"steps":[x["steps"] for x in rows],"episodes":rows})
        successes=sum(int(x["success"]) for x in episodes); result={"classification":"PASS","experiment":c["experiment_id"],"execution_pass":True,"policy_success_count":successes,"policy_success_total":len(episodes),"policy_success_rate":successes/len(episodes),"task_success_count":sum(int(x["success_count"]>0) for x in task_summaries),"task_total":len(task_summaries),"task_summaries":task_summaries,"total_environment_steps":sum(x["steps"] for x in episodes),"model_predictions":predictions,"model_load_s":ready["model_load_s"],"total_inference_s":float(np.sum(all_inf)),"mean_inference_s":float(np.mean(all_inf)),"median_inference_s":float(np.median(all_inf)),"p95_inference_s":float(np.percentile(all_inf,95)),"peak_vram_allocated_bytes":latest["peak_vram_allocated_bytes"],"peak_vram_reserved_bytes":latest["peak_vram_reserved_bytes"],"worker_rss_kib":latest["worker_rss_kib_at_result"],"coordinator_rss_kib":proc.memory_info().rss//1024,"elapsed_s":time.perf_counter()-started,"checkpoint_revision":ready["checkpoint_revision"],"trajectory":{"path":trajectory.relative_to(ROOT).as_posix(),"sha256":sha256(trajectory)},"boundary":c["boundary"]}; result_path.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2),flush=True)
    finally:
        if env is not None: env.close()
        if worker is not None and worker.poll() is None: worker.terminate(); worker.wait(10)

if __name__=="__main__": main()
