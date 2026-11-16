#!/usr/bin/env python3
import hashlib,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; C=ROOT/"configs/task/phase6d_feagine_a03_short_dynamics_v1.yaml"; S=ROOT/"scripts/run_phase6d_feagine_a03_short_dynamics.py"; R=ROOT/"results/phase6d_feagine_a03_short_dynamics_v1_static.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 c=yaml.safe_load(C.read_text()); text=S.read_text(); checks={"phase6c_pass":yaml.safe_load((ROOT/"results/phase6c_feagine_a03_runtime_zero_step_v2.yaml").read_text())["classification"]=="PASS","bounded_steps":c["protocol"]["steps"]==200 and c["protocol"]["simulated_duration_s"]==1.0,"headless_cpu":not c["authorization"]["gui"] and not c["authorization"]["render"] and not c["authorization"]["gpu"],"no_policy_hardware":not c["authorization"]["policy_inference"] and not c["authorization"]["hardware"],"fresh_targets":all(not (ROOT/c[k]).exists() for k in ["output_dir","structured_result","raw_log"]),"exact_step_loop":text.count("mujoco.mj_step(model,data)")==1,"no_runtime_imports":"mujoco" not in sys.modules}; out={"classification":"PASS" if all(checks.values()) else "FAIL","experiment":"phase6d_feagine_a03_short_dynamics_v1_static","checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"source_hashes":{"config":sha(C),"runner":sha(S)},"environment_steps":0,"boundary":"Static only; no runtime imports, MJCF load, simulation, rendering, policy or hardware."}; R.write_text(yaml.safe_dump(out,sort_keys=False)); print(yaml.safe_dump(out,sort_keys=False)); raise SystemExit(0 if out["classification"]=="PASS" else 1)
if __name__=="__main__": main()
