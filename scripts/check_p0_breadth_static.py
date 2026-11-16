#!/usr/bin/env python3
"""Read-only/static contract check for the P0-Breadth 5-task gate."""
from pathlib import Path
import json, os, sys, yaml
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_CONFIG_PATH",str(ROOT/"configs/libero/phase2a_runtime"))
from libero.libero import benchmark
cfg=yaml.safe_load((ROOT/"configs/eval/p0_breadth_openvla_base_10task30_v2.yaml").read_text())
suite=benchmark.get_benchmark_dict()[cfg["suite"]["name"]](task_order_index=cfg["suite"]["task_order_index"])
assert [x["task_id"] for x in cfg["tasks"]]==list(range(10))
assert cfg["protocol"]["init_state_indices"]==[0,1,2] and cfg["protocol"]["total_episodes"]==30
for item in cfg["tasks"]:
    task=suite.get_task(item["task_id"]); assert task.name==item["name"] and task.language==item["instruction"]
for key in ("output_dir","structured_result","raw_log"): assert not (ROOT/cfg[key]).exists(),(key,cfg[key])
source=(ROOT/"scripts/evaluate_p0_breadth_openvla_base.py").read_text()
for token in ("refusing overwrite","task_summaries","checkpoint_revision","trajectory","policy_success_count"):
    assert token in source,token
print(json.dumps({"classification":"PASS","tasks":len(cfg["tasks"]),"episodes":30,"task_ids":list(range(10)),"init_states":[0,1,2],"fresh_paths":True,"cuda_calls":0,"environment_steps":0},indent=2))
