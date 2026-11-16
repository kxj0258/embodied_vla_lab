#!/usr/bin/env python3
"""Read-only Phase 6B asset-contract verifier."""
import hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; RESULT=ROOT/"results/phase6b_feagine_a03_asset_contract_v1.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=yaml.safe_load(RESULT.read_text()); checks={"pass":r["classification"]=="PASS" and r["checks_passed"]==19,"decision":r["decision"]=="ASSET_CONTRACT_PASS_RUNTIME_INSTALL_PENDING","wheel":sha(ROOT/"feagine-simulation/dist/feagine_mujoco-0.1.2-py3-none-any.whl")==r["source_hashes"]["wheel"],"config":sha(ROOT/"configs/task/phase6b_feagine_a03_asset_contract_v1.yaml")==r["source_hashes"]["config"],"adapter":sha(ROOT/"scripts/feagine_a03_bridge.py")==r["source_hashes"]["adapter"],"no_runtime":not r["packages_installed"] and not r["mjcf_loaded"] and r["environment_steps"]==0 and r["hardware_commands"]==0}
 out={"classification":"PASS" if all(checks.values()) else "FAIL","checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"cuda_initialized":False,"environment_steps":0}; print(yaml.safe_dump(out,sort_keys=False)); raise SystemExit(0 if out["classification"]=="PASS" else 1)
if __name__=="__main__": main()
