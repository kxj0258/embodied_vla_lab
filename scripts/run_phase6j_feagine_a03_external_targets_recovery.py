#!/usr/bin/env python3
import hashlib,json,math,time
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]; C=ROOT/"configs/task/phase6j_feagine_a03_external_targets_recovery_dev4_v1.yaml"
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
 start=time.perf_counter(); episodes=[]
 for spec in c["targets"]:
  model=load_mujoco_model(preset_id=c["preset_id"],model_type="mjcf"); data=mujoco.MjData(model); robot=FeagineMjcfRobot(model,data,preset_id=c["preset_id"]); mujoco.mj_forward(model,data); initial=np.asarray(robot.tip_pose().position,dtype=np.float64); base=np.asarray(robot.grasper_base_pose().position,dtype=np.float64); direction=initial-base; direction/=np.linalg.norm(direction); target=initial+np.asarray(spec["delta_m"],dtype=np.float64)
  opts=core.InverseKinematicsOptions(); opts.max_iterations=c["ik"]["max_iterations"]; opts.position_tolerance=c["ik"]["position_tolerance_m"]; opts.direction_tolerance=c["ik"]["direction_tolerance"]; opts.clamp_theta=True; opts.theta_limit=c["ik"]["theta_limit_rad"]; seed=[core.PccSectionConfig(.16,spec["seed_angles_rad"][i],spec["seed_angles_rad"][i+1]) for i in range(0,6,2)]; solved=robot.solve_inverse_kinematics(target,direction,options=opts,initial_guess=seed,apply_solution=False); raw=np.asarray([v for s in solved.solved_sections for v in (float(s.theta),float(s.phi))]); applied=canonical(raw); robot.drive_section_angles(applied); robot.set_grip_command(c["protocol"]["gripper_command"]); robot.drive_grasper_rotation(c["protocol"]["grasper_rotation_rad"])
  limited=[]
  for j in range(model.njnt):
   if model.jnt_limited[j]: limited.append((int(model.jnt_qposadr[j]),int(model.jnt_type[j]),*map(float,model.jnt_range[j])))
  finite=True; max_h=max_s=0.; contacts=0
  for _ in range(c["protocol"]["steps_per_target"]):
   mujoco.mj_step(model,data); finite=finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.ctrl).all()); contacts=max(contacts,int(data.ncon))
   for adr,typ,lo,hi in limited:
    q=float(data.qpos[adr]); pen=max(lo-q,q-hi,0.); max_h=max(max_h,pen) if typ==int(mujoco.mjtJoint.mjJNT_HINGE) else max_h; max_s=max(max_s,pen) if typ==int(mujoco.mjtJoint.mjJNT_SLIDE) else max_s
  final=np.asarray(robot.tip_pose().position,dtype=np.float64); ie=float(np.linalg.norm(initial-target)); fe=float(np.linalg.norm(final-target)); progress=(ie-fe)/ie; material=max_h>c["diagnostic_thresholds"]["hinge_material_violation_rad"] or max_s>c["diagnostic_thresholds"]["slide_material_violation_m"]; checks={"ik_success":bool(solved.success),"canonical_theta_nonnegative":bool(np.all(applied[0::2]>=0)),"finite_state":finite,"exact_time":abs(float(data.time)-2.)<1e-12,"terminal_error":fe<=c["per_target_gate"]["terminal_position_error_m"],"progress":progress>=c["per_target_gate"]["minimum_progress_fraction"],"no_material_limit_violation":not material}; episodes.append({"target_id":spec["id"],"seed_id":spec["seed_id"],"classification":"PASS" if all(checks.values()) else "FAIL","checks":checks,"raw_angles_rad":raw.tolist(),"applied_angles_rad":applied.tolist(),"initial_error_m":ie,"final_error_m":fe,"progress_fraction":progress,"ik_iterations":int(solved.iterations),"ik_position_error_m":float(solved.final_position_error),"ik_direction_error":float(solved.final_direction_error),"max_hinge_penetration_rad":max_h,"max_slide_penetration_m":max_s,"max_contacts":contacts})
 successes=sum(x["classification"]=="PASS" for x in episodes); passed=successes==c["aggregate_gate"]["required_successes"]; result={"classification":"PASS" if passed else "FAIL","experiment":c["experiment_id"],"decision":"REACHABILITY_CORRECTED_DEV4_PASS" if passed else "REACHABILITY_CORRECTED_DEV4_FAIL","successes":successes,"total_targets":len(episodes),"total_steps":1600,"simulated_time_total_s":8.0,"episodes":episodes,"elapsed_s":time.perf_counter()-start,"cuda_initialized":False,"model_predictions":0,"reward":None,"hardware_commands":0,"source_hashes":{"config":sha(C),"phase6f":sha(ROOT/"results/phase6f_feagine_a03_external_targets_dev4_v1.yaml"),"phase6i":sha(ROOT/"results/phase6i_feagine_a03_reachability_scan_recovery_v2.yaml")},"boundary":c["boundary"]}; out.mkdir(parents=True); (out/"result.json").write_text(json.dumps(result,indent=2)+"\n"); rp.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2)); raise SystemExit(0 if passed else 1)
if __name__=="__main__": main()
