#!/usr/bin/env python3
import hashlib,json,time
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]; CONFIG=ROOT/"configs/task/phase6g_feagine_a03_mapping_diagnostic_v1.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pose_axis(pose):
 w,x,y,z=np.asarray(pose.quaternion,dtype=np.float64); r=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]); return r@np.array([0.,0.,1.])
def cosine(a,b):
 na=float(np.linalg.norm(a)); nb=float(np.linalg.norm(b)); return float(np.dot(a,b)/(na*nb)) if na>1e-12 and nb>1e-12 else None
def main():
 c=yaml.safe_load(CONFIG.read_text()); out=ROOT/c["output_dir"]; rp=ROOT/c["structured_result"]
 for p in (out,rp):
  if p.exists(): raise FileExistsError(f"refusing overwrite: {p}")
 import mujoco,pyfeagine_sim_core as core
 from feagine_mujoco import FeagineMjcfRobot,load_mujoco_model
 start=time.perf_counter(); rows=[]
 for spec in c["targets"]:
  model=load_mujoco_model(preset_id=c["preset_id"],model_type="mjcf"); data=mujoco.MjData(model); robot=FeagineMjcfRobot(model,data,preset_id=c["preset_id"]); mujoco.mj_forward(model,data)
  initial=np.asarray(robot.tip_pose().position,dtype=np.float64); base=np.asarray(robot.grasper_base_pose().position,dtype=np.float64); direction=initial-base; direction/=np.linalg.norm(direction); desired=np.asarray(spec["delta_m"],dtype=np.float64); target=initial+desired
  opts=core.InverseKinematicsOptions(); opts.max_iterations=c["ik"]["max_iterations"]; opts.position_tolerance=c["ik"]["position_tolerance_m"]; opts.direction_tolerance=c["ik"]["direction_tolerance"]; opts.clamp_theta=True; opts.theta_limit=c["ik"]["theta_limit_rad"]
  solved=robot.solve_inverse_kinematics(target,direction,options=opts,apply_solution=False); raw=np.array([v for s in solved.solved_sections for v in (float(s.theta),float(s.phi))],dtype=np.float64)
  # Preserve Phase 6F behavior exactly: apply the returned solution even when solver success is false.
  robot.solve_inverse_kinematics(target,direction,options=opts,apply_solution=True); applied=np.asarray(robot.get_section_angles(),dtype=np.float64); controls=np.asarray(data.ctrl,dtype=np.float64).copy(); pcc_controls=controls[:48].copy()
  fk=robot.forward_kinematics(); end=fk.section_end_poses[-1]; pred_dir=pose_axis(end); pred_dir/=np.linalg.norm(pred_dir); tool_len=float(np.linalg.norm(np.asarray(robot.preset.grasper_tool_tip_position,dtype=np.float64))); core_pred=np.asarray(end.position,dtype=np.float64)+tool_len*pred_dir; core_disp=core_pred-initial
  finite=True
  for _ in range(c["protocol"]["steps_per_target"]):
   mujoco.mj_step(model,data); finite=finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.ctrl).all())
  final=np.asarray(robot.tip_pose().position,dtype=np.float64); actual=final-initial; desired_norm=float(np.linalg.norm(desired)); changed=np.abs(applied-raw); rows.append({"target_id":spec["id"],"ik_success":bool(solved.success),"ik_iterations":int(solved.iterations),"ik_position_error_m":float(solved.final_position_error),"ik_direction_error":float(solved.final_direction_error),"raw_solved_section_angles_rad":raw.tolist(),"applied_section_angles_rad":applied.tolist(),"angle_change_abs_rad":changed.tolist(),"angle_change_max_rad":float(changed.max()),"changed_angle_count":int(np.sum(changed>c["diagnostic_thresholds"]["angle_change_rad"])),"pcc_actuator_target_count":int(pcc_controls.size),"pcc_actuator_targets":pcc_controls.tolist(),"pcc_actuator_min":float(pcc_controls.min()),"pcc_actuator_max":float(pcc_controls.max()),"desired_displacement_m":desired.tolist(),"core_predicted_displacement_m":core_disp.tolist(),"actual_displacement_m":actual.tolist(),"desired_vs_core_cosine":cosine(desired,core_disp),"desired_vs_actual_cosine":cosine(desired,actual),"core_vs_actual_cosine":cosine(core_disp,actual),"core_displacement_gain":float(np.linalg.norm(core_disp)/desired_norm),"actual_displacement_gain":float(np.linalg.norm(actual)/desired_norm),"final_error_m":float(np.linalg.norm(final-target)),"finite_state":finite,"simulated_time_s":float(data.time)})
 altered=sum(x["changed_angle_count"]>0 for x in rows); direction_bad=sum(x["desired_vs_actual_cosine"] is None or x["desired_vs_actual_cosine"]<c["diagnostic_thresholds"]["direction_cosine_mismatch"] for x in rows); gain_bad=sum(x["actual_displacement_gain"]<c["diagnostic_thresholds"]["displacement_gain_low"] or x["actual_displacement_gain"]>c["diagnostic_thresholds"]["displacement_gain_high"] for x in rows)
 result={"classification":"PASS" if all(x["finite_state"] and abs(x["simulated_time_s"]-2.0)<1e-12 and x["pcc_actuator_target_count"]==48 for x in rows) else "FAIL","experiment":c["experiment_id"],"decision":"MAPPING_DIAGNOSTIC_COMPLETE","source_phase6f_result":"FAIL_0_OF_4","targets_diagnosed":len(rows),"targets_with_angle_change":altered,"targets_with_direction_mismatch":direction_bad,"targets_with_gain_mismatch":gain_bad,"episodes":rows,"elapsed_s":time.perf_counter()-start,"cuda_initialized":False,"model_predictions":0,"reward":None,"hardware_commands":0,"source_hashes":{"config":sha(CONFIG),"phase6f_result":sha(ROOT/"results/phase6f_feagine_a03_external_targets_dev4_v1.yaml")},"boundary":c["boundary"]}; out.mkdir(parents=True); (out/"result.json").write_text(json.dumps(result,indent=2)+"\n"); rp.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["classification"]=="PASS" else 1)
if __name__=="__main__": main()
