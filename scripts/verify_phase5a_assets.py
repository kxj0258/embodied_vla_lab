#!/usr/bin/env python3
"""Read-only verifier for Phase 5A data-contract evidence."""
import hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; MANIFEST=ROOT/"results/phase5a_world_model_data_v1_manifest.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    m=yaml.safe_load(MANIFEST.read_text()); checks={"classification":m["classification"]=="PASS","visual_qa":m["visual_qa"]=="PASS"}; s=m["source"]; sp=ROOT/s["path"]; checks["source_hash"]=sp.is_file() and sha(sp)==s["sha256"]
    for name,item in m["artifacts"].items():
        p=ROOT/item["path"]; checks[f"{name}_hash"]=p.is_file() and p.stat().st_size==item["bytes"] and sha(p)==item["sha256"]
    classification="PASS" if all(checks.values()) else "FAIL"; print(yaml.safe_dump({"classification":classification,"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"cuda_initialized":False,"environment_steps":0,"model_predictions":0},sort_keys=False)); raise SystemExit(0 if classification=="PASS" else 1)
if __name__=="__main__": main()
