#!/usr/bin/env python3
import hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; M=ROOT/"results/phase6g_feagine_a03_mapping_diagnostic_v1_manifest.yaml"
def main():
 m=yaml.safe_load(M.read_text()); checks=[]
 for item in m["artifacts"]:
  b=(ROOT/item["path"]).read_bytes(); checks += [len(b)==item["bytes"],hashlib.sha256(b).hexdigest()==item["sha256"]]
 r=yaml.safe_load((ROOT/"results/phase6g_feagine_a03_mapping_diagnostic_v1.yaml").read_text()); e=r["episodes"]
 checks += [r["classification"]=="PASS",r["decision"]=="MAPPING_DIAGNOSTIC_COMPLETE",r["source_phase6f_result"]=="FAIL_0_OF_4",r["targets_diagnosed"]==4,r["targets_with_angle_change"]==2,r["targets_with_gain_mismatch"]==4,[x["ik_success"] for x in e]==[False,False,True,True],e[2]["changed_angle_count"]==1 and e[3]["changed_angle_count"]==2,e[2]["core_vs_actual_cosine"]>.999 and e[3]["core_vs_actual_cosine"]>.999,all(x["pcc_actuator_target_count"]==48 for x in e),r["model_predictions"]==0 and r["hardware_commands"]==0,r["reward"] is None]
 out={"classification":"PASS" if all(checks) else "FAIL","checks_passed":sum(checks),"checks_total":len(checks)}; print(yaml.safe_dump(out,sort_keys=False),end=""); raise SystemExit(0 if all(checks) else 1)
if __name__=="__main__": main()
