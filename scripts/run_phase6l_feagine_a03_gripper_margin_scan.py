#!/usr/bin/env python3
import hashlib,json,math,time
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]; C=ROOT/"configs/task/phase6l_feagine_a03_gripper_margin_scan_v1.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def wrap(v): return (v+math.pi)%(2*math.pi)-math.pi
def canonical(v):
 out=[]
 for theta,phi in zip(v[0::2],v[1::2]):
  if theta<0: theta=-theta; phi=wrap(phi+math.pi)
  else: phi=wrap(phi)
  out.extend([theta,phi])
 return np.asarray(out,dtype=np.float64)
def main():
 c=yaml.safe_load(C.read_text()); out=ROOT/c["output_dir"]; rp=ROOT/c["structured_result"]
 for p in (out,rp):
  if p.exists(): raise FileExistsError(f"refusing overwrite: {p}")
 import mujoco,pyfeagine_sim_core as core
 from feagine_mujoco import FeagineMjcfRobot,load_mujoco_model
 started=time.perf_counter(); source=yaml.safe_load((ROOT/"results/phase6j_feagine_a03_external_targets_recovery_dev4_v1.yaml").read_text()); rows=[{"gripper_command":0.0,"source":"phase6j_reused","successes":source["successes"],"total_targets":source["total_targets"],"strict_pass":source["successes"]==4,"episodes":[{"target_id":x["target_id"],"classification":x["classification"],"final_error_m":x["final_error_m"],"progress_fraction":x["progress_fraction"],"max_slide_penetration_m":x["max_slide_penetration_m"]} for x in source["episodes"]]}]
 for command in c["execution_commands"]:
  episodes=[]
  for spec in c["targets"]:
   model=load_mujoco_model(preset_id=c["preset_id"],model_type="mjcf"); data=mujoco.MjData(model); robot=FeagineMjcfRobot(model,data,preset_id=c["preset_id"]); mujoco.mj_forward(model,data); initial=np.asarray(robot.tip_pose().position,dtype=np.float64); base=np.asarray(robot.grasper_base_pose().position,dtype=np.float64); direction=initial-base; direction/=np.linalg.norm(direction); target=initial+np.asarray(spec["delta_m"],dtype=np.float64)
   opts=core.InverseKinematicsOptions(); opts.max_iterations=c["ik"]["max_iterations"]; opts.position_tolerance=c["ik"]["position_tolerance_m"]; opts.direction_tolerance=c["ik"]["direction_tolerance"]; opts.clamp_theta=True; opts.theta_limit=c["ik"]["theta_limit_rad"]; seed=[core.PccSectionConfig(.16,spec["seed_angles_rad"][i],spec["seed_angles_rad"][i+1]) for i in range(0,6,2)]; solved=robot.solve_inverse_kinematics(target,direction,options=opts,initial_guess=seed,apply_solution=False); raw=np.asarray([v for s in solved.solved_sections for v in (float(s.theta),float(s.phi))]); robot.drive_section_angles(canonical(raw)); robot.set_grip_command(command); robot.drive_grasper_rotation(c["protocol"]["grasper_rotation_rad"]); slide=[]
   for j in range(model.njnt):
    if model.jnt_limited[j] and int(model.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_SLIDE): slide.append((int(model.jnt_qposadr[j]),*map(float,model.jnt_range[j])))
   finite=True; max_slide=0.
   for _ in range(c["protocol"]["steps_per_target"]):
    mujoco.mj_step(model,data); finite=finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.ctrl).all())
    for adr,lo,hi in slide:
     q=float(data.qpos[adr]); max_slide=max(max_slide,lo-q,q-hi,0.)
   final=np.asarray(robot.tip_pose().position,dtype=np.float64); ie=float(np.linalg.norm(initial-target)); fe=float(np.linalg.norm(final-target)); progress=(ie-fe)/ie; checks={"ik_success":bool(solved.success),"finite_state":finite,"exact_time":abs(float(data.time)-2.)<1e-12,"terminal_error":fe<=c["per_target_gate"]["terminal_position_error_m"],"progress":progress>=c["per_target_gate"]["minimum_progress_fraction"],"no_material_slide_violation":max_slide<=c["per_target_gate"]["material_slide_threshold_m"]}; episodes.append({"target_id":spec["id"],"classification":"PASS" if all(checks.values()) else "FAIL","checks":checks,"final_error_m":fe,"progress_fraction":progress,"max_slide_penetration_m":max_slide})
  successes=sum(x["classification"]=="PASS" for x in episodes); rows.append({"gripper_command":command,"source":"new_execution","successes":successes,"total_targets":len(episodes),"strict_pass":successes==c["aggregate_gate"]["required_successes"],"episodes":episodes})
 candidates=[x for x in rows if x["strict_pass"]]; selected=min((x["gripper_command"] for x in candidates),default=None); execution_pass=all(all(ep["checks"]["finite_state"] and ep["checks"]["exact_time"] for ep in row["episodes"]) for row in rows if row["source"]=="new_execution"); result={"classification":"PASS" if execution_pass else "FAIL","experiment":c["experiment_id"],"decision":"MINIMUM_SAFE_GRIPPER_MARGIN_SELECTED" if selected is not None and execution_pass else "NO_SAFE_GRIPPER_MARGIN_SELECTED","execution_pass":execution_pass,"selected_gripper_command":selected,"selection_rule":c["selection_rule"],"command_results":rows,"new_targets_run":12,"total_steps":4800,"simulated_time_total_s":24.0,"elapsed_s":time.perf_counter()-started,"cuda_initialized":False,"model_predictions":0,"reward":None,"hardware_commands":0,"source_hashes":{"config":sha(C),"phase6j":sha(ROOT/"results/phase6j_feagine_a03_external_targets_recovery_dev4_v1.yaml"),"phase6k":sha(ROOT/"results/phase6k_feagine_a03_slide_limit_diagnostic_v1.yaml")},"boundary":c["boundary"]}; out.mkdir(parents=True); (out/"result.json").write_text(json.dumps(result,indent=2)+"\n"); rp.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["classification"]=="PASS" else 1)
if __name__=="__main__": main()
