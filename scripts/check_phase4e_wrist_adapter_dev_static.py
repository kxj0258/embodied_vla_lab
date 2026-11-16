#!/usr/bin/env python3
"""CPU-only static contract checker for Phase 4E."""
import hashlib, sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/eval/phase4e_openvla_wrist_adapter_dev3_v1.yaml"
ENTRY=ROOT/"scripts/evaluate_phase4e_openvla_wrist_adapter_dev.py"
WORKER=ROOT/"scripts/run_phase4e_openvla_wrist_adapter_worker.py"
BASE=ROOT/"results/phase4b_openvla_viewpoint_dev30_v1.yaml"
ADAPTER=ROOT/"outputs/train/phase4c_openvla_wrist_qlora_rank8_train20_val10_v1_seed20260829/adapter/adapter_model.safetensors"
RESULT=ROOT/"results/phase4e_openvla_wrist_adapter_task7_dev3_v1_static.yaml"

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    c=yaml.safe_load(CONFIG.read_text()); b=yaml.safe_load(BASE.read_text()); rows=[x for x in b["episodes"] if x["task_id"]==7 and x["view"]=="eye_in_hand"]
    checks={
      "canonical_revision":c["model"]["revision"]=="fa5ae1e7509348889295bba8e08621d8b55e9baf",
      "task7_development_only":c["task"]["task_id"]==7 and c["task"]["name"]=="turn_on_the_stove",
      "three_frozen_states":c["protocol"]["init_state_indices"]==[0,1,2] and c["protocol"]["paired_seeds"]==[20261120,20261121,20261122],
      "base_rows_exact":[(x["init_state_index"],x["paired_seed"],x["success"],x["steps"]) for x in rows]==[(0,20261120,False,600),(1,20261121,False,600),(2,20261122,False,600)],
      "eye_in_hand_only":c["protocol"]["observation_key"]=="eye_in_hand",
      "adapter_hash":digest(ADAPTER)==c["model"]["adapter_sha256"]=="fad1e3201dacc5e1136f45b5df3b06943d83354a34c1079d620c4b1abcf1ae7a",
      "evaluator_frozen":digest(ENTRY)=="62cbab76681f4c383d312264cc83b2cd2253c5a037dec51ead08b31d3368087e",
      "worker_wrapper_frozen":digest(WORKER)=="2c7541247d6a344c1db1ba013a2c62e1fb721b61af7b0efceae5fb9f95aaea53",
      "decision_rule_frozen":c["decision_rule"]=={"adapter_successes_0":"NO_GO_NO_CLOSED_LOOP_RECOVERY","adapter_successes_1_or_2":"DEVELOPMENT_RECOVERY_SIGNAL_PARTIAL","adapter_successes_3":"DEVELOPMENT_RECOVERY_SIGNAL_STRONG","automatic_final_authorization":False},
      "fresh_targets":not (ROOT/c["output_dir"]).exists() and not (ROOT/c["structured_result"]).exists() and not (ROOT/c["raw_log"]).exists(),
      "final_not_authorized":c["decision_rule"]["automatic_final_authorization"] is False and "authorize final tasks" in c["boundary"],
      "no_cuda_initialized":"torch" not in sys.modules,
    }
    classification="PASS" if all(checks.values()) else "FAIL"
    result={"classification":classification,"experiment":"phase4e_openvla_wrist_adapter_task7_dev3_v1_static","checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"cuda_initialized":False,"environment_resets":0,"environment_steps":0,"model_predictions":0,"boundary":"Static validation only; no CUDA, environment, model, training, success or reward evidence."}
    if RESULT.exists(): raise FileExistsError(f"refusing overwrite: {RESULT}")
    RESULT.write_text(yaml.safe_dump(result,sort_keys=False)); print(yaml.safe_dump(result,sort_keys=False)); raise SystemExit(0 if classification=="PASS" else 1)

if __name__=="__main__": main()
