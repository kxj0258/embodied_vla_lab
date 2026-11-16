#!/usr/bin/env python3
"""Static validation for the corrected Phase 4A camera gate v2."""
from __future__ import annotations
import ast, hashlib, json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/eval/phase4a_openvla_viewpoint_camera_gate_v2.yaml"
ENTRY=ROOT/"scripts/probe_phase4a_libero_viewpoints_v2.py"
IMPL=ROOT/"scripts/probe_phase4a_libero_viewpoints.py"
RESULT=ROOT/"results/phase4a_openvla_viewpoint_camera_gate_v2_static.yaml"
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    c=yaml.safe_load(CONFIG.read_text()); ast.parse(ENTRY.read_text()); ast.parse(IMPL.read_text())
    p=c["protocol"]; s=c["splits_preregistered_for_later_rollouts"]
    checks={
      "new_v2_id": c["experiment_id"]=="phase4a_openvla_viewpoint_camera_gate_v2_seed20261101",
      "v1_failure_preserved": (ROOT/c["supersedes_failed_run"]).is_file(),
      "observation_alias_correct": p["observation_keys"]["eye_in_hand"]=="eye_in_hand",
      "mujoco_camera_corrected": p["simulator_camera_names"]["eye_in_hand"]=="robot0_eye_in_hand",
      "agentview_unchanged": p["simulator_camera_names"]["agentview"]=="agentview",
      "zero_step_prediction": p["environment_steps"]==p["model_predictions"]==0,
      "frozen_revision": c["model"]["revision"]=="fa5ae1ed633f94762dfdd28a0a305b643b2ba8baf",
      "split_unchanged": s["development_task_ids"]==[0,3,6,7,8] and s["final_task_ids"]==[1,2,4,5,9],
      "fresh_targets": all(not (ROOT/c[k]).exists() for k in ["output_dir","structured_result","raw_log"]),
    }
    result={"classification":"PASS" if all(checks.values()) else "FAIL","checks":checks,
      "config_sha256":digest(CONFIG),"entrypoint_sha256":digest(ENTRY),"implementation_sha256":digest(IMPL),
      "cuda_calls":0,"environment_resets":0,"environment_steps":0,"model_predictions":0,
      "boundary":c["boundary"]}
    if RESULT.exists(): raise FileExistsError(f"refusing overwrite: {RESULT}")
    RESULT.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2))
    if result["classification"]!="PASS": raise SystemExit(1)
if __name__=="__main__": main()
