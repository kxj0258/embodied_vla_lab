#!/usr/bin/env python3
"""Read-only checkpoint/result/figure verifier for Phase 5C."""
import hashlib
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]; RESULT=ROOT/"results/phase5c_world_model_train100_pair_v1.yaml"; MANIFEST=ROOT/"results/phase5c_world_model_smoke_v1_manifest.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=yaml.safe_load(RESULT.read_text()); m=yaml.safe_load(MANIFEST.read_text()); checks={"result_pass":r["classification"]=="PASS","decision_no_go":r["decision"]=="ENGINEERING_PASS_OFFLINE_BASELINE_NO_GO" and r["formal_training_go"] is False,"manifest_pass":m["classification"]=="PASS" and m["visual_qa"]=="PASS"}
 for name,item in r["artifacts"].items():
  p=ROOT/item["path"]; checks[f"{name}_hash"]=p.is_file() and p.stat().st_size==item["bytes"] and sha(p)==item["sha256"]
 for name,item in m["artifacts"].items():
  p=ROOT/item["path"]; checks[f"figure_{name}_hash"]=p.is_file() and p.stat().st_size==item["bytes"] and sha(p)==item["sha256"]
 for name in ["action_free_checkpoint","action_conditioned_checkpoint"]:
  ck=torch.load(ROOT/r["artifacts"][name]["path"],map_location="cpu",weights_only=False); checks[f"{name}_state"]=ck["step"]==100 and len(ck["optimizer"]["state"])>0 and all(torch.isfinite(x).all() for x in ck["model"].values())
 classification="PASS" if all(checks.values()) else "FAIL"; print(yaml.safe_dump({"classification":classification,"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"cuda_initialized":False,"environment_steps":0,"checkpoint_selected":False},sort_keys=False)); raise SystemExit(0 if classification=="PASS" else 1)
if __name__=="__main__": main()
