#!/usr/bin/env python3
import hashlib, sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
C=ROOT/"configs/task/phase6e_feagine_a03_move_to_target_smoke_v1.yaml"
S=ROOT/"scripts/run_phase6e_feagine_a03_move_to_target_smoke.py"
R=ROOT/"results/phase6e_feagine_a03_move_to_target_smoke_v1_static.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 c=yaml.safe_load(C.read_text()); text=S.read_text(); gate=c["success_gate"]
 checks={"phase6d_v2_pass":yaml.safe_load((ROOT/"results/phase6d_feagine_a03_limit_diagnostic_v2.yaml").read_text())["decision"]=="SOFT_LIMIT_PENETRATION_WITHIN_DIAGNOSTIC_THRESHOLD","small_core_target":c["target_generator"]["section_angles_rad"]==[.06,0,.04,0,.02,0],"bounded_steps":c["protocol"]["steps"]==400 and c["protocol"]["timestep_s"]==.005 and text.count("mujoco.mj_step(model, data)")==1,"frozen_success_gate":gate=={"terminal_position_error_m":.015,"minimum_progress_fraction":.80,"minimum_target_displacement_m":.02,"maximum_target_displacement_m":.08},"frozen_material_thresholds":c["diagnostic_thresholds"]=={"hinge_material_violation_rad":.001,"slide_material_violation_m":.0001},"fresh_targets":all(not (ROOT/c[k]).exists() for k in ["output_dir","structured_result","raw_log"]),"headless_cpu_no_policy":all(c["authorization"][k] is False for k in ["gui","render","gpu","policy_inference","hardware"]),"records_ik_and_progress":all(x in text for x in ["initial_error_m","final_error_m","progress_fraction","solved_section_angles_rad"]),"fail_closed_material":'"no_material_limit_violation": not material' in text,"no_runtime_imports":"mujoco" not in sys.modules and not any(x.startswith("torch") for x in sys.modules)}
 out={"classification":"PASS" if all(checks.values()) else "FAIL","experiment":"phase6e_feagine_a03_move_to_target_smoke_v1_static","checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"source_hashes":{"config":sha(C),"runner":sha(S)},"environment_steps":0,"boundary":"Static authorization only; no MuJoCo load/step, GPU, policy, rendering, or hardware."}; R.write_text(yaml.safe_dump(out,sort_keys=False)); print(yaml.safe_dump(out,sort_keys=False)); raise SystemExit(0 if out["classification"]=="PASS" else 1)
if __name__=="__main__": main()
