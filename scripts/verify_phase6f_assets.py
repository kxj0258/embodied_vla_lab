#!/usr/bin/env python3
import hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; M=ROOT/"results/phase6f_feagine_a03_external_targets_dev4_v1_manifest.yaml"
def main():
 m=yaml.safe_load(M.read_text()); checks=[]
 for item in m["artifacts"]:
  data=(ROOT/item["path"]).read_bytes(); checks += [len(data)==item["bytes"],hashlib.sha256(data).hexdigest()==item["sha256"]]
 r=yaml.safe_load((ROOT/"results/phase6f_feagine_a03_external_targets_dev4_v1.yaml").read_text()); eps=r["episodes"]
 checks += [r["classification"]=="FAIL",r["decision"]=="EXTERNAL_TARGETS_DEV4_FAIL",r["successes"]==0 and r["total_targets"]==4,r["total_steps"]==1600,all(x["checks"]["finite_state"] and x["checks"]["exact_time"] for x in eps),all(x["checks"]["no_material_limit_violation"] for x in eps),[x["checks"]["ik_success"] for x in eps]==[False,False,True,True],r["model_predictions"]==0 and r["hardware_commands"]==0,r["reward"] is None]
 out={"classification":"PASS" if all(checks) else "FAIL","checks_passed":sum(checks),"checks_total":len(checks)}; print(yaml.safe_dump(out,sort_keys=False),end=""); raise SystemExit(0 if all(checks) else 1)
if __name__=="__main__": main()
