#!/usr/bin/env python3
import hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; M=ROOT/"results/phase6k_feagine_a03_slide_limit_diagnostic_v1_manifest.yaml"
def main():
 m=yaml.safe_load(M.read_text()); checks=[]
 for x in m["artifacts"]:
  b=(ROOT/x["path"]).read_bytes(); checks += [len(b)==x["bytes"],hashlib.sha256(b).hexdigest()==x["sha256"]]
 r=yaml.safe_load((ROOT/"results/phase6k_feagine_a03_slide_limit_diagnostic_v1.yaml").read_text()); e=r["episodes"]
 checks += [r["classification"]=="PASS",r["source_phase6j_result"]=="FAIL_2_OF_4",r["targets_diagnosed"]==4,e[0]["material_violations"][0]["name"]=="feagine_grasper_joint_0",e[1]["material_violations"][0]["name"]=="feagine_grasper_joint_1",e[0]["material_violations"][0]["peak_violation_step"]==1 and e[1]["material_violations"][0]["peak_violation_step"]==1,e[0]["material_violations"][0]["min_m"]<-.013 and e[1]["material_violations"][0]["min_m"]<-.013,r["paired_summary"]["x_material_violation_counts"]==[1,1],r["paired_summary"]["y_material_violation_counts"]==[0,0],r["model_predictions"]==0 and r["hardware_commands"]==0,r["reward"] is None]
 out={"classification":"PASS" if all(checks) else "FAIL","checks_passed":sum(checks),"checks_total":len(checks)}; print(yaml.safe_dump(out,sort_keys=False),end=""); raise SystemExit(0 if all(checks) else 1)
if __name__=="__main__": main()
