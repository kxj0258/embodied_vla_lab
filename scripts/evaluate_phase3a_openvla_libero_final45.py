#!/usr/bin/env python3
"""Preregistered 45-state OpenVLA base-policy final with one persistent model load."""
from __future__ import annotations
import json, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled"); os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0"); os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/evla_phase3a_final45_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/evla_phase3a_final45_matplotlib"); sys.dont_write_bytecode = True

import numpy as np
import psutil, yaml
from PIL import Image
from lerobot.envs.libero import LiberoEnv
from libero.libero import benchmark

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_phase3a_openvla_libero_single_episode import read_worker_event, scalar  # noqa: E402
from phase3a_openvla_bridge_protocol import SCHEMA_VERSION,load_json,sha256,validate_response,write_json_atomic  # noqa: E402
CONFIG=ROOT/'configs/eval/openvla_libero_phase3a_final_heldout45.yaml'


def main():
    if os.environ.get('LIBERO_CONFIG_PATH') != str(ROOT/'configs/libero/phase2a_runtime'): raise ValueError('frozen LIBERO_CONFIG_PATH required')
    if (Path.home()/'.libero/config.yaml').exists(): raise RuntimeError('refusing user-home LIBERO config')
    cfg=yaml.safe_load(Path(os.environ.get('EVLA_FINAL45_CONFIG',CONFIG)).read_text())
    indices=[int(x) for x in cfg['protocol']['init_state_indices']]
    if indices != list(range(5,50)) or cfg['protocol']['development_excluded_init_state_indices'] != list(range(5)) or cfg['protocol']['episodes'] != 45: raise ValueError('final protocol mismatch')
    if cfg['protocol']['schema_version'] != SCHEMA_VERSION: raise ValueError('schema mismatch')
    output=ROOT/cfg['output_dir']
    if output.exists(): raise FileExistsError(f'refusing overwrite: {output}')
    output.mkdir(parents=True); ipc=output/'ipc'; ipc.mkdir(); (output/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    seed0=int(cfg['protocol']['evaluator_seed_start']); random.seed(seed0); np.random.seed(seed0)
    suite=benchmark.get_benchmark_dict()[cfg['task']['suite']](task_order_index=0); task_id=int(cfg['task']['task_id'])
    if suite.get_task(task_id).name != cfg['task']['name']: raise ValueError('task mismatch')
    env=LiberoEnv(task_suite=suite,task_id=task_id,task_suite_name=cfg['task']['suite'],episode_length=600,
        camera_name=['agentview_image','robot0_eye_in_hand_image'],camera_name_mapping={'agentview_image':'agentview','robot0_eye_in_hand_image':'eye_in_hand'},
        obs_type='pixels_agent_pos',observation_width=128,observation_height=128,init_states=True,episode_index=5,n_envs=1,
        num_steps_wait=5,control_freq=20,control_mode='relative',hard_reset=True)
    worker=None; started=time.perf_counter(); process=psutil.Process(); trajectory_path=output/'trajectory.jsonl'; latest=None
    try:
        worker=subprocess.Popen([cfg['model']['openvla_python'],str(ROOT/cfg['model']['worker'])],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'})
        ready=read_worker_event(worker,int(cfg['protocol']['worker_start_timeout_s']))
        if ready.get('event')!='ready' or ready.get('classification')!='PASS': raise RuntimeError('worker not ready')
        episodes=[]; inference=[]; predictions=0; instruction=cfg['task']['instruction']; prompt=f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        with trajectory_path.open('x') as trajectory:
            for episode,init_index in enumerate(indices):
                obs,_=env.reset(seed=seed0+episode); reward_sum=0.; max_reward=float('-inf'); success=False; steps=0; ep_inf=0.; term=False; trunc=False; ep_start=time.perf_counter()
                epdir=ipc/f'episode_{episode:02d}_init_{init_index:02d}'; epdir.mkdir()
                for step in range(1,601):
                    image=np.asarray(obs['pixels']['agentview'])
                    if image.shape!=(128,128,3) or image.dtype!=np.uint8: raise ValueError('invalid image')
                    model_image=image[::-1,::-1].copy(); image_path=epdir/f'step_{step:04d}_agentview_rot180.png'; Image.fromarray(model_image,mode='RGB').save(image_path)
                    request_path=epdir/f'step_{step:04d}_request.json'; response_path=epdir/f'step_{step:04d}_response.json'
                    request={'schema_version':SCHEMA_VERSION,'experiment_id':cfg['experiment_id'],'request_id':f"{cfg['experiment_id']}:episode:{episode:02d}:step:{step:04d}",
                        'checkpoint_revision':cfg['model']['revision'],'unnorm_key':cfg['model']['unnorm_key'],'instruction':instruction,'prompt':prompt,'image_path':image_path.name,
                        'image_sha256':sha256(image_path),'image_shape':[128,128,3],'image_dtype':'uint8','preprocess':cfg['protocol']['model_image_preprocess'],'environment_steps_requested':1}
                    write_json_atomic(request_path,request); worker.stdin.write(json.dumps({'command':'predict','request':str(request_path),'response':str(response_path)})+'\n'); worker.stdin.flush()
                    event=read_worker_event(worker,int(cfg['protocol']['per_action_timeout_s']))
                    if event.get('event')!='prediction' or event.get('request_id')!=request['request_id']: raise RuntimeError('prediction identity mismatch')
                    response=load_json(response_path); latest=response
                    if response['request_sha256']!=sha256(request_path): raise ValueError('request hash mismatch')
                    action=validate_response(response,request); obs,reward,terminated,truncated,info=env.step(action); steps=step; predictions+=1
                    rv=float(scalar(reward)); reward_sum+=rv; max_reward=max(max_reward,rv); success=success or bool(info.get('is_success',False)); term=bool(scalar(terminated)); trunc=bool(scalar(truncated))
                    inf=float(response['inference_s']); inference.append(inf); ep_inf+=inf
                    row={'episode':episode,'init_state_index':init_index,'step':step,'reward':rv,'success':success,'terminated':term,'truncated':trunc,'inference_s':inf,'action':response['libero_adapted_action'],'image_sha256':request['image_sha256'],'response_sha256':sha256(response_path)}
                    trajectory.write(json.dumps(row)+'\n'); trajectory.flush()
                    if term or trunc: break
                summary={'episode':episode,'init_state_index':init_index,'seed':seed0+episode,'success':success,'steps':steps,'sum_reward':reward_sum,'max_reward':max_reward,'terminated':term,'truncated':trunc,'inference_s':ep_inf,'episode_wall_s':time.perf_counter()-ep_start}
                episodes.append(summary); print(json.dumps({'episode_summary':summary}),flush=True)
        worker.stdin.write(json.dumps({'command':'shutdown'})+'\n'); worker.stdin.flush(); shutdown=read_worker_event(worker,30)
        if shutdown.get('event')!='shutdown' or shutdown.get('predictions')!=predictions: raise RuntimeError('shutdown mismatch')
        worker.wait(timeout=30)
        if worker.returncode!=0 or latest is None: raise RuntimeError('worker final state invalid')
        successes=sum(int(x['success']) for x in episodes)
        result={'classification':'PASS','experiment_id':cfg['experiment_id'],'execution_pass':True,'episodes':episodes,'success_count':successes,'success_total':45,'success_rate':successes/45,
            'mean_sum_reward':float(np.mean([x['sum_reward'] for x in episodes])),'mean_max_reward':float(np.mean([x['max_reward'] for x in episodes])),'total_environment_steps':sum(x['steps'] for x in episodes),
            'model_predictions':predictions,'model_load_s':ready['model_load_s'],'total_inference_s':float(np.sum(inference)),'mean_inference_s':float(np.mean(inference)),'median_inference_s':float(np.median(inference)),
            'p95_inference_s':float(np.percentile(inference,95)),'min_inference_s':float(np.min(inference)),'max_inference_s':float(np.max(inference)),'rollout_end_to_end_s':time.perf_counter()-started,
            'passive_vram_allocated_bytes':ready['passive_vram_allocated_bytes'],'passive_vram_reserved_bytes':ready['passive_vram_reserved_bytes'],'peak_vram_allocated_bytes':latest['peak_vram_allocated_bytes'],
            'peak_vram_reserved_bytes':latest['peak_vram_reserved_bytes'],'worker_rss_kib_at_result':latest['worker_rss_kib_at_result'],'coordinator_rss_kib_at_result':process.memory_info().rss//1024,
            'trajectory_sha256':sha256(trajectory_path),'boundary':cfg['boundary']}
        (output/'eval_info.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try: worker.wait(timeout=10)
            except subprocess.TimeoutExpired: worker.kill(); worker.wait(timeout=10)
        env.close(); print(json.dumps({'cleanup':'env_closed_and_worker_reaped'}),flush=True)


if __name__=='__main__': main()
