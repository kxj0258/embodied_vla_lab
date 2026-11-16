#!/usr/bin/env python3
import hashlib, json, time
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/task/phase6f_feagine_a03_external_targets_dev4_v1.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 c=yaml.safe_load(CONFIG.read_text()); output=ROOT/c["output_dir"]; result_path=ROOT/c["structured_result"]
 for p in (output,result_path):
  if p.exists(): raise FileExistsError(f"refusing overwrite: {p}")
 import mujoco
 import pyfeagine_sim_core as core
 from feagine_mujoco import FeagineMjcfRobot,load_mujoco_model
 start=time.perf_counter(); episodes=[]
 for spec in c["targets"]:
  model=load_mujoco_model(preset_id=c["preset_id"],model_type="mjcf"); data=mujoco.MjData(model); robot=FeagineMjcfRobot(model,data,preset_id=c["preset_id"]); mujoco.mj_forward(model,data)
  initial=np.asarray(robot.tip_pose().position,dtype=np.float64); grasper=np.asarray(robot.grasper_base_pose().position,dtype=np.float64); direction=initial-grasper; direction/=np.linalg.norm(direction); target=initial+np.asarray(spec["delta_m"],dtype=np.float64)
  opts=core.InverseKinematicsOptions(); opts.max_iterations=c["ik"]["max_iterations"]; opts.position_tolerance=c["ik"]["position_tolerance_m"]; opts.direction_tolerance=c["ik"]["direction_tolerance"]; opts.clamp_theta=True; opts.theta_limit=c["ik"]["theta_limit_rad"]
  solved=robot.solve_inverse_kinematics(target,direction,options=opts,apply_solution=True); robot.set_grip_command(c["protocol"]["gripper_command"]); robot.drive_grasper_rotation(c["protocol"]["grasper_rotation_rad"])
  limited=[]
  for j in range(model.njnt):
   if model.jnt_limited[j]: limited.append((int(model.jnt_qposadr[j]),int(model.jnt_type[j]),*map(float,model.jnt_range[j])))
  max_h=max_s=0.0; finite=True; max_contacts=0
  for _ in range(c["protocol"]["steps_per_target"]):
   mujoco.mj_step(model,data); finite=finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.ctrl).all()); max_contacts=max(max_contacts,int(data.ncon))
   for adr,typ,lo,hi in limited:
    q=float(data.qpos[adr]); pen=max(lo-q,q-hi,0.0)
    if typ==int(mujoco.mjtJoint.mjJNT_HINGE): max_h=max(max_h,pen)
    if typ==int(mujoco.mjtJoint.mjJNT_SLIDE): max_s=max(max_s,pen)
  final=np.asarray(robot.tip_pose().position,dtype=np.float64); initial_error=float(np.linalg.norm(initial-target)); final_error=float(np.linalg.norm(final-target)); progress=(initial_error-final_error)/initial_error
  material=max_h>c["diagnostic_thresholds"]["hinge_material_violation_rad"] or max_s>c["diagnostic_thresholds"]["slide_material_violation_m"]
  checks={"ik_success":bool(solved.success),"finite_state":finite,"exact_time":abs(float(data.time)-c["protocol"]["steps_per_target"]*c["protocol"]["timestep_s"])<1e-12,"terminal_error":final_error<=c["per_target_gate"]["terminal_position_error_m"],"progress":progress>=c["per_target_gate"]["minimum_progress_fraction"],"no_material_limit_violation":not material}
  episodes.append({"target_id":spec["id"],"delta_m":spec["delta_m"],"classification":"PASS" if all(checks.values()) else "FAIL","checks":checks,"target_position_m":target.tolist(),"final_tip_position_m":final.tolist(),"initial_error_m":initial_error,"final_error_m":final_error,"progress_fraction":float(progress),"ik_iterations":int(solved.iterations),"ik_position_error_m":float(solved.final_position_error),"ik_direction_error":float(solved.final_direction_error),"max_hinge_penetration_rad":max_h,"max_slide_penetration_m":max_s,"max_contacts":max_contacts})
 successes=sum(x["classification"]=="PASS" for x in episodes); aggregate=successes==c["aggregate_gate"]["required_successes"] and len(episodes)==c["aggregate_gate"]["total_targets"]
 result={"classification":"PASS" if aggregate else "FAIL","experiment":c["experiment_id"],"decision":"EXTERNAL_TARGETS_DEV4_PASS" if aggregate else "EXTERNAL_TARGETS_DEV4_FAIL","successes":successes,"total_targets":len(episodes),"total_steps":len(episodes)*c["protocol"]["steps_per_target"],"simulated_time_total_s":sum(c["protocol"]["steps_per_target"]*c["protocol"]["timestep_s"] for _ in episodes),"episodes":episodes,"elapsed_s":time.perf_counter()-start,"cuda_initialized":False,"model_predictions":0,"reward":None,"hardware_commands":0,"source_hashes":{"config":sha(CONFIG)},"boundary":c["boundary"]}
 output.mkdir(parents=True); (output/"result.json").write_text(json.dumps(result,indent=2)+"\n"); result_path.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2)); raise SystemExit(0 if aggregate else 1)
if __name__=="__main__": main()
