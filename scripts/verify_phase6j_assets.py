#!/usr/bin/env python3
import hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; M=ROOT/"results/phase6j_feagine_a03_external_targets_recovery_dev4_v1_manifest.yaml"
def main():
 m=yaml.safe_load(M.read_text()); checks=[]
 for x in m["artifacts"]:
  b=(ROOT/x["path"]).read_bytes(); checks += [len(b)==x["bytes"],hashlib.sha256(b).hexdigest()==x["sha256"]]
 r=yaml.safe_load((ROOT/"results/phase6j_feagine_a03_external_targets_recovery_dev4_v1.yaml").read_text()); e=r["episodes"]
 checks += [r["classification"]=="FAIL",r["successes"]==2 and r["total_targets"]==4,all(x["checks"]["ik_success"] for x in e),all(x["checks"]["terminal_error"] and x["checks"]["progress"] for x in e),[x["checks"]["no_material_limit_violation"] for x in e]==[False,False,True,True],all(x["max_hinge_penetration_rad"]==0 for x in e),e[0]["max_slide_penetration_m"]>.0001 and e[1]["max_slide_penetration_m"]>.0001,e[2]["max_slide_penetration_m"]<.0001 and e[3]["max_slide_penetration_m"]<.0001,all(x["max_contacts"]==0 for x in e),r["model_predictions"]==0 and r["hardware_commands"]==0,r["reward"] is None]
 out={"classification":"PASS" if all(checks) else "FAIL","checks_passed":sum(checks),"checks_total":len(checks)}; print(yaml.safe_dump(out,sort_keys=False),end=""); raise SystemExit(0 if all(checks) else 1)
if __name__=="__main__": main()
