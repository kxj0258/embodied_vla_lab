#!/usr/bin/env python3
"""Read-only integrity verifier for canonical Phase 5E evidence."""
import hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; MANIFEST=ROOT/"results/phase5e_residual_world_model_dev_summary_v1_manifest.yaml"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 m=yaml.safe_load(MANIFEST.read_text()); result=yaml.safe_load((ROOT/m["integrity"]["canonical_result"]["path"]).read_text()); checks={}
 for name,item in m["integrity"].items():
  path=ROOT/item["path"]; checks[f"{name}_exists"]=path.is_file(); checks[f"{name}_size_hash"]=path.stat().st_size==item["bytes"] and sha(path)==item["sha256"]
 checks["execution_pass"]=result["classification"]=="PASS" and result["execution_pass"] is True
 checks["decision_no_go"]=result["decision"]=="DEVELOPMENT_NO_GO_ACTION_CONDITIONING"
 h4=m["development_h4_pixel_mae_0_255"]; checks["go_rule_failed"]=h4["residual_action_conditioned"]>=h4["residual_action_free"] and h4["residual_action_conditioned"]>=h4["copy_current_frame"]
 checks["boundary_preserved"]=result["prior_exposed_confirmation_executed"] is False and result["checkpoint_selected"] is False and result["environment_steps"]==0
 payload={"classification":"PASS" if all(checks.values()) else "FAIL","checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"cuda_initialized":False,"environment_steps":0}; print(yaml.safe_dump(payload,sort_keys=False)); raise SystemExit(0 if payload["classification"]=="PASS" else 1)
if __name__=="__main__": main()
