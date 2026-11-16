#!/usr/bin/env python3
import hashlib,json,math,time
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]; C=ROOT/"configs/task/phase6k_feagine_a03_slide_limit_diagnostic_v1.yaml"
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
 started=time.perf_counter(); episodes=[]
 for spec in c["targets"]:
  model=load_mujoco_model(preset_id=c["preset_id"],model_type="mjcf"); data=mujoco.MjData(model); robot=FeagineMjcfRobot(model,data,preset_id=c["preset_id"]); mujoco.mj_forward(model,data); initial=np.asarray(robot.tip_pose().position,dtype=np.float64); base=np.asarray(robot.grasper_base_pose().position,dtype=np.float64); direction=initial-base; direction/=np.linalg.norm(direction); target=initial+np.asarray(spec["delta_m"],dtype=np.float64)
  opts=core.InverseKinematicsOptions(); opts.max_iterations=c["ik"]["max_iterations"]; opts.position_tolerance=c["ik"]["position_tolerance_m"]; opts.direction_tolerance=c["ik"]["direction_tolerance"]; opts.clamp_theta=True; opts.theta_limit=c["ik"]["theta_limit_rad"]; seed=[core.PccSectionConfig(.16,spec["seed_angles_rad"][i],spec["seed_angles_rad"][i+1]) for i in range(0,6,2)]; solved=robot.solve_inverse_kinematics(target,direction,options=opts,initial_guess=seed,apply_solution=False); raw=np.asarray([v for s in solved.solved_sections for v in (float(s.theta),float(s.phi))]); robot.drive_section_angles(canonical(raw)); robot.set_grip_command(c["protocol"]["gripper_command"]); robot.drive_grasper_rotation(c["protocol"]["grasper_rotation_rad"])
  slides=[]
  for j in range(model.njnt):
   if model.jnt_limited[j] and int(model.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_SLIDE):
    adr=int(model.jnt_qposadr[j]); lo,hi=map(float,model.jnt_range[j]); q=float(data.qpos[adr]); slides.append({"name":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j),"qpos_adr":adr,"lower_m":lo,"upper_m":hi,"min_m":q,"max_m":q,"first_violation_step":None,"peak_violation_step":None,"max_penetration_m":0.0})
  finite=True
  for step in range(1,c["protocol"]["steps_per_target"]+1):
   mujoco.mj_step(model,data); finite=finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.ctrl).all())
   for x in slides:
    q=float(data.qpos[x["qpos_adr"]]); x["min_m"]=min(x["min_m"],q); x["max_m"]=max(x["max_m"],q); pen=max(x["lower_m"]-q,q-x["upper_m"],0.0)
    if pen>c["diagnostic"]["nominal_violation_epsilon_m"] and x["first_violation_step"] is None: x["first_violation_step"]=step
    if pen>x["max_penetration_m"]: x["max_penetration_m"]=pen; x["peak_violation_step"]=step
  violated=[x for x in slides if x["max_penetration_m"]>c["diagnostic"]["nominal_violation_epsilon_m"]]; material=[x for x in slides if x["max_penetration_m"]>c["diagnostic"]["material_slide_threshold_m"]]; final=np.asarray(robot.tip_pose().position,dtype=np.float64); episodes.append({"target_id":spec["id"],"finite_state":finite,"simulated_time_s":float(data.time),"final_error_m":float(np.linalg.norm(final-target)),"slide_joint_count":len(slides),"violating_slide_joint_count":len(violated),"material_violating_slide_joint_count":len(material),"max_slide_penetration_m":max((x["max_penetration_m"] for x in slides),default=0.0),"material_violations":material,"all_slide_joints":slides})
 result={"classification":"PASS" if len(episodes)==4 and all(x["finite_state"] and abs(x["simulated_time_s"]-2.)<1e-12 for x in episodes) else "FAIL","experiment":c["experiment_id"],"decision":"SLIDE_LIMIT_DIAGNOSTIC_COMPLETE","source_phase6j_result":"FAIL_2_OF_4","targets_diagnosed":len(episodes),"episodes":episodes,"paired_summary":{"x_material_violation_counts":[episodes[0]["material_violating_slide_joint_count"],episodes[1]["material_violating_slide_joint_count"]],"y_material_violation_counts":[episodes[2]["material_violating_slide_joint_count"],episodes[3]["material_violating_slide_joint_count"]]},"elapsed_s":time.perf_counter()-started,"cuda_initialized":False,"model_predictions":0,"reward":None,"hardware_commands":0,"source_hashes":{"config":sha(C),"phase6j":sha(ROOT/"results/phase6j_feagine_a03_external_targets_recovery_dev4_v1.yaml")},"boundary":c["boundary"]}; out.mkdir(parents=True); (out/"result.json").write_text(json.dumps(result,indent=2)+"\n"); rp.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["classification"]=="PASS" else 1)
if __name__=="__main__": main()
