#!/usr/bin/env python3
import hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; M=ROOT/"results/phase6h_feagine_signed_theta_multistart_static_v1_manifest.yaml"
def main():
 m=yaml.safe_load(M.read_text()); checks=[]
 for x in m["artifacts"]:
  b=(ROOT/x["path"]).read_bytes(); checks += [len(b)==x["bytes"],hashlib.sha256(b).hexdigest()==x["sha256"]]
 r=yaml.safe_load((ROOT/"results/phase6h_feagine_signed_theta_multistart_static_v1.yaml").read_text()); rows=r["target_solutions"]
 best=[min(a["canonical_position_error_m"] for a in row["attempts"]) for row in rows]
 checks += [r["classification"]=="FAIL",r["checks_passed"]==4 and r["checks_total"]==5,r["checks"]["fk_position_equivalence"],r["checks"]["fk_direction_equivalence"],not r["checks"]["all_targets_solved"],r["equivalence_samples"]==260,r["max_fk_position_difference_m"]<=1e-6,r["max_fk_direction_difference"]<=1e-6,all(row["valid_seed_count"]==0 for row in rows),all(.0015<x<.0017 for x in best),r["environment_steps"]==0 and r["model_predictions"]==0 and r["hardware_commands"]==0]
 out={"classification":"PASS" if all(checks) else "FAIL","checks_passed":sum(checks),"checks_total":len(checks)}; print(yaml.safe_dump(out,sort_keys=False),end=""); raise SystemExit(0 if all(checks) else 1)
if __name__=="__main__": main()
