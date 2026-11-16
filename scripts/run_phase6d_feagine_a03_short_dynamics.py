#!/usr/bin/env python3
"""Bounded 200-step headless A03 Type-2 dynamics smoke."""
import hashlib,json,time
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]; C=ROOT/"configs/task/phase6d_feagine_a03_short_dynamics_v1.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 c=yaml.safe_load(C.read_text()); out=ROOT/c["output_dir"]; result_path=ROOT/c["structured_result"]
 for p in [out,result_path]:
  if p.exists(): raise FileExistsError(f"refusing overwrite: {p}")
 import mujoco
 from feagine_mujoco import FeagineMjcfRobot,find_terminal_capsule_body_name,load_mujoco_model
 start=time.perf_counter(); model=load_mujoco_model(preset_id=c["preset_id"],model_type="mjcf"); data=mujoco.MjData(model); robot=FeagineMjcfRobot(model,data,preset_id=c["preset_id"]); mujoco.mj_forward(model,data); body_name=find_terminal_capsule_body_name(model); body_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body_name); initial=np.asarray(data.xpos[body_id]).copy(); robot.drive_section_angles(c["protocol"]["section_angles"]); robot.set_grip_command(c["protocol"]["gripper_command"]); robot.drive_grasper_rotation(c["protocol"]["grasper_rotation_rad"])
 max_abs_qvel=0.; max_abs_ctrl=float(np.max(np.abs(data.ctrl))); max_contacts=0
 for _ in range(c["protocol"]["steps"]):
  mujoco.mj_step(model,data); max_abs_qvel=max(max_abs_qvel,float(np.max(np.abs(data.qvel)))); max_abs_ctrl=max(max_abs_ctrl,float(np.max(np.abs(data.ctrl)))); max_contacts=max(max_contacts,int(data.ncon))
 final=np.asarray(data.xpos[body_id]).copy(); finite=all(np.isfinite(x).all() for x in [data.qpos,data.qvel,data.ctrl,data.xpos]); limited=[]
 for j in range(model.njnt):
  if model.jnt_limited[j]:
   adr=model.jnt_qposadr[j]; lo,hi=model.jnt_range[j]; limited.append(bool(lo-1e-6<=data.qpos[adr]<=hi+1e-6))
 motion=float(np.linalg.norm(final-initial)); expected=c["protocol"]["steps"]*model.opt.timestep; checks={"finite_state":finite,"exact_time":abs(float(data.time)-expected)<1e-12,"tip_motion":motion>=c["checks"]["require_tip_motion_min_m"],"joint_limits":all(limited),"step_budget":c["protocol"]["steps"]==200,"headless":not c["authorization"]["gui"] and not c["authorization"]["render"],"no_policy_or_hardware":not c["authorization"]["policy_inference"] and not c["authorization"]["hardware"]}; checks={k:bool(v) for k,v in checks.items()}; result={"classification":"PASS" if all(checks.values()) else "FAIL","experiment":c["experiment_id"],"decision":"SHORT_DYNAMICS_PASS_MOVE_TO_TARGET_GATE_ALLOWED" if all(checks.values()) else "SHORT_DYNAMICS_FAIL","checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"steps":c["protocol"]["steps"],"simulated_time_s":float(data.time),"tip_body":body_name,"tip_initial_m":initial.tolist(),"tip_final_m":final.tolist(),"tip_motion_m":motion,"max_abs_qvel":max_abs_qvel,"max_abs_ctrl":max_abs_ctrl,"max_contacts":max_contacts,"limited_joints_checked":len(limited),"elapsed_s":time.perf_counter()-start,"steps_per_s":c["protocol"]["steps"]/(time.perf_counter()-start),"cuda_initialized":False,"model_predictions":0,"reward":None,"success":None,"hardware_commands":0,"source_hashes":{"config":sha(C)},"boundary":c["boundary"]}; out.mkdir(parents=True); (out/"result.json").write_text(json.dumps(result,indent=2)+"\n"); result_path.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["classification"]=="PASS" else 1)
if __name__=="__main__": main()
