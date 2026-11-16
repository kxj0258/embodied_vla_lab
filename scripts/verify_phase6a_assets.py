#!/usr/bin/env python3
"""Read-only Phase 6A evidence verifier."""
import hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 paths={"v1_expected":ROOT/"results/phase6a_feagine_bridge_static_v1_failure.yaml","v2_expected":ROOT/"results/phase6a_feagine_bridge_static_v2_failure.yaml","v3_fail":ROOT/"results/phase6a_feagine_bridge_static_v3.yaml","v4_pass":ROOT/"results/phase6a_feagine_bridge_static_v4.yaml"}; docs={k:yaml.safe_load(p.read_text()) for k,p in paths.items()}
 checks={"all_evidence_exists":all(p.is_file() for p in paths.values()),"v1_expected":docs["v1_expected"]["classification"]=="EXPECTED_EXCEPTION","v2_expected":docs["v2_expected"]["classification"]=="EXPECTED_EXCEPTION","v3_fail":docs["v3_fail"]["classification"]=="FAIL" and docs["v3_fail"]["checks_passed"]==13,"v4_pass":docs["v4_pass"]["classification"]=="PASS" and docs["v4_pass"]["checks_passed"]==14,"roundtrip":docs["v4_pass"]["roundtrip"]["samples"]==256 and docs["v4_pass"]["roundtrip"]["max_position_error_m"]<1e-9,"no_execution":all(d.get("environment_steps",d.get("scope",{}).get("environment_steps"))==0 for d in docs.values()),"source_hashes":docs["v4_pass"]["source_hashes"]["implementation"]==sha(ROOT/"scripts/feagine_bridge.py") and docs["v4_pass"]["source_hashes"]["config"]==sha(ROOT/"configs/task/phase6a_feagine_bridge_static_v4.yaml")}
 payload={"classification":"PASS" if all(checks.values()) else "FAIL","checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"cuda_initialized":False,"environment_steps":0,"hardware_commands":0}; print(yaml.safe_dump(payload,sort_keys=False)); raise SystemExit(0 if payload["classification"]=="PASS" else 1)
if __name__=="__main__": main()
