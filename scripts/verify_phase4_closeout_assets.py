#!/usr/bin/env python3
"""Read-only verifier for frozen Phase 4 closeout evidence and figures."""
import hashlib
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
MANIFESTS=[
 ROOT/"results/phase4b_viewpoint_dev_summary_v2_manifest.yaml",
 ROOT/"results/phase4d_wrist_fullval_summary_v1_manifest.yaml",
 ROOT/"results/phase4e_wrist_no_go_summary_v1_manifest.yaml",
 ROOT/"results/phase4_viewpoint_closeout_v1_manifest.yaml",
]
RESULTS=[
 ROOT/"results/phase4a_openvla_viewpoint_camera_gate_v2.yaml",
 ROOT/"results/phase4b_openvla_viewpoint_dev30_v1.yaml",
 ROOT/"results/phase4d_openvla_wrist_step20_vs_base_fullval889_paired.yaml",
 ROOT/"results/phase4e_openvla_wrist_adapter_task7_dev3_v1.yaml",
]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def records(m):
    for key in ("artifact","artifacts"):
        value=m.get(key)
        if isinstance(value,dict) and "path" in value: yield value
        elif isinstance(value,dict):
            for item in value.values():
                if isinstance(item,dict) and "path" in item: yield item
    for item in m.get("outputs",[]):
        if isinstance(item,dict) and "path" in item: yield item
def main():
    checks={"all_results_present":all(p.is_file() for p in RESULTS),"all_manifests_present":all(p.is_file() for p in MANIFESTS)}
    artifact_count=0
    for mp in MANIFESTS:
        m=yaml.safe_load(mp.read_text()); checks[f"{mp.stem}_classification"]=m["classification"]=="PASS"; checks[f"{mp.stem}_visual_qa"]=m.get("visual_qa")=="PASS"
        for item in records(m):
            p=ROOT/item["path"]; artifact_count+=1; checks[f"artifact_{artifact_count}_hash"]=p.is_file() and p.stat().st_size==item["bytes"] and sha(p)==item["sha256"]
    classification="PASS" if all(checks.values()) else "FAIL"
    print(yaml.safe_dump({"classification":classification,"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"artifact_count":artifact_count,"cuda_initialized":False,"environment_steps":0,"model_predictions":0},sort_keys=False))
    raise SystemExit(0 if classification=="PASS" else 1)
if __name__=="__main__": main()
