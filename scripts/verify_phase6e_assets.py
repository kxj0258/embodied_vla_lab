#!/usr/bin/env python3
import hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; M=ROOT/"results/phase6e_feagine_a03_move_to_target_smoke_v1_manifest.yaml"
def main():
 m=yaml.safe_load(M.read_text()); checks=[]
 for item in m["artifacts"]:
  data=(ROOT/item["path"]).read_bytes(); checks += [len(data)==item["bytes"], hashlib.sha256(data).hexdigest()==item["sha256"]]
 r=yaml.safe_load((ROOT/"results/phase6e_feagine_a03_move_to_target_smoke_v1.yaml").read_text())
 checks += [r["classification"]=="PASS",r["decision"]=="BOUNDED_CORE_TARGET_REACHING_PASS",r["checks_passed"]==r["checks_total"]==9,r["steps"]==400,abs(r["simulated_time_s"]-2.0)<1e-12,r["final_error_m"]<=.015,r["progress_fraction"]>=.80,r["max_hinge_penetration_rad"]<=.001,r["max_slide_penetration_m"]<=.0001,r["model_predictions"]==0 and r["hardware_commands"]==0,r["reward"] is None]
 out={"classification":"PASS" if all(checks) else "FAIL","checks_passed":sum(checks),"checks_total":len(checks)}; print(yaml.safe_dump(out,sort_keys=False),end=""); raise SystemExit(0 if all(checks) else 1)
if __name__=="__main__": main()
