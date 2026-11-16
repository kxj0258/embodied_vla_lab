#!/usr/bin/env python3
"""CUDA-free static contract check for Phase 4B paired viewpoint development."""
from __future__ import annotations
import ast, hashlib, json, os
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; os.environ.setdefault("LIBERO_CONFIG_PATH",str(ROOT/"configs/libero/phase2a_runtime"))
from libero.libero import benchmark
CONFIG=ROOT/"configs/eval/phase4b_openvla_viewpoint_dev30_v1.yaml"; EVALUATOR=ROOT/"scripts/evaluate_phase4b_openvla_viewpoint_dev.py"; RESULT=ROOT/"results/phase4b_openvla_viewpoint_dev30_v1_static.yaml"
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    c=yaml.safe_load(CONFIG.read_text()); p=c["protocol"]; source=EVALUATOR.read_text(); ast.parse(source)
    suite=benchmark.get_benchmark_dict()[c["suite"]["name"]](task_order_index=c["suite"]["task_order_index"])
    tasks_ok=all(suite.get_task(x["task_id"]).name==x["name"] and suite.get_task(x["task_id"]).language==x["instruction"] for x in c["tasks"])
    checks={"development_tasks_exact":[x["task_id"] for x in c["tasks"]]==[0,3,6,7,8],"metadata_exact":tasks_ok,"views_exact":c["view_conditions"]==[{"name":"agentview","observation_key":"agentview"},{"name":"eye_in_hand","observation_key":"eye_in_hand"}],"init_states_exact":p["init_state_indices"]==[0,1,2],"episode_arithmetic":len(c["tasks"])*len(p["init_state_indices"])*len(c["view_conditions"])==p["total_episodes"]==30,"paired_seed_count":p["paired_state_count"]==15,"canonical_revision":c["model"]["revision"]=="fa5ae1e7509348889295bba8e08621d8b55e9baf","fresh_targets":all(not (ROOT/c[k]).exists() for k in ["output_dir","structured_result","raw_log"]),"persistent_model_single_load":"worker=subprocess.Popen" in source and source.count("worker=subprocess.Popen")==1,"fresh_env_per_episode":"episode_index=int(init_index)" in source and "env.close(); env=None" in source,"view_selected_only_at_image":"obs[\"pixels\"][obs_key]" in source,"paired_statistics_present":all(x in source for x in ["paired_outcome_counts","agentview_only","eye_in_hand_only","wilson_95"]),"no_training_code":all(x not in source for x in ["optimizer.step","loss.backward","PeftModel"]),"trajectory_and_cleanup":all(x in source for x in ["trajectory.jsonl","shutdown","worker.terminate"])}
    result={"classification":"PASS" if all(checks.values()) else "FAIL","experiment_id":c["experiment_id"],"checks":checks,"config_sha256":digest(CONFIG),"evaluator_sha256":digest(EVALUATOR),"cuda_calls":0,"environment_resets":0,"environment_steps":0,"model_predictions":0,"development_tasks":[0,3,6,7,8],"final_tasks_untouched":[1,2,4,5,9],"paired_states":15,"episodes":30,"boundary":c["boundary"]}
    if RESULT.exists(): raise FileExistsError(f"refusing overwrite: {RESULT}")
    RESULT.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2))
    if result["classification"]!="PASS": raise SystemExit(1)
if __name__=="__main__": main()
