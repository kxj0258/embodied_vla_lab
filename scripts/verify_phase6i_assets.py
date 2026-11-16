#!/usr/bin/env python3
import hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; M=ROOT/"results/phase6i_feagine_a03_reachability_summary_v1_manifest.yaml"
def main():
 m=yaml.safe_load(M.read_text()); checks=[]
 for x in m["artifacts"]:
  b=(ROOT/x["path"]).read_bytes(); checks += [len(b)==x["bytes"],hashlib.sha256(b).hexdigest()==x["sha256"]]
 v1=yaml.safe_load((ROOT/"results/phase6i_feagine_a03_reachability_scan_v1.yaml").read_text()); v2=yaml.safe_load((ROOT/"results/phase6i_feagine_a03_reachability_scan_recovery_v2.yaml").read_text())
 checks += [v1["classification"]=="FAIL" and v1["checks_passed"]==5,v2["classification"]=="PASS" and v2["checks_passed"]==10,v2["decision"]=="REACHABILITY_SCAN_EVIDENCE_RECOVERED",all(x==.0015 for x in v2["minimum_retractions_m"].values()),v2["minimum_retraction_spread_m"]==0.0,v2["reused_ik_attempts"]==672 and v2["new_ik_attempts"]==0,v2["environment_steps"]==0]
 out={"classification":"PASS" if all(checks) else "FAIL","checks_passed":sum(checks),"checks_total":len(checks)}; print(yaml.safe_dump(out,sort_keys=False),end=""); raise SystemExit(0 if all(checks) else 1)
if __name__=="__main__": main()
