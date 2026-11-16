#!/usr/bin/env python3
import hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; M=ROOT/"results/phase6c_feagine_a03_runtime_summary_v1_manifest.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=yaml.safe_load(M.read_text()); r=yaml.safe_load((ROOT/m["canonical_result"]["path"]).read_text()); checks={"result_hash":sha(ROOT/m["canonical_result"]["path"])==m["canonical_result"]["sha256"],"log_hash":sha(ROOT/m["raw_log"]["path"])==m["raw_log"]["sha256"],"internal_hash":sha(ROOT/m["internal_result"]["path"])==m["internal_result"]["sha256"],"pass":r["classification"]=="PASS" and r["checks_passed"]==11,"zero_step":r["environment_steps"]==0 and r["simulated_time_s"]==0 and r["runtime"]["data_time_before_after"]==[0.0,0.0],"v1_fail_retained":yaml.safe_load((ROOT/"results/phase6c_feagine_a03_runtime_zero_step_v1.yaml").read_text())["classification"]=="FAIL"}
 out={"classification":"PASS" if all(checks.values()) else "FAIL","checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"cuda_initialized":False,"environment_steps":0}; print(yaml.safe_dump(out,sort_keys=False)); raise SystemExit(0 if out["classification"]=="PASS" else 1)
if __name__=="__main__": main()
