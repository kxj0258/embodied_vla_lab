#!/usr/bin/env python3
"""Read-only integrity verifier for Phase 5B static evidence."""
import hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; RESULT=ROOT/"results/phase5b_small_world_model_static_v1.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    r=yaml.safe_load(RESULT.read_text()); checks={"classification":r["classification"]=="PASS","all_checks":r["checks_passed"]==r["checks_total"]==16,"config_hash":sha(ROOT/"configs/train/phase5b_small_world_model_static_v1.yaml")==r["source_hashes"]["config"],"model_hash":sha(ROOT/"scripts/phase5_small_world_model.py")==r["source_hashes"]["model_implementation"],"cpu_only":r["cuda_initialized"] is False,"no_training":r["training_steps"]==0 and r["optimizer_steps"]==0,"no_environment":r["environment_steps"]==0 and r["environment_resets"]==0}
    classification="PASS" if all(checks.values()) else "FAIL"; print(yaml.safe_dump({"classification":classification,"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"cuda_initialized":False,"environment_steps":0},sort_keys=False)); raise SystemExit(0 if classification=="PASS" else 1)
if __name__=="__main__": main()
