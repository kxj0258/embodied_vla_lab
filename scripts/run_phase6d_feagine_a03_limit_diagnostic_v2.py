#!/usr/bin/env python3
import hashlib,json,time
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]; C=ROOT/"configs/task/phase6d_feagine_a03_limit_diagnostic_v2.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 c=yaml.safe_load(C.read_text()); out=ROOT/c["output_dir"]; rp=ROOT/c["structured_result"]
 for p in [out,rp]:
  if p.exists(): raise FileExistsError(f"refusing overwrite: {p}")
 import mujoco
 from feagine_mujoco import FeagineMjcfRobot,load_mujoco_model
 start=time.perf_counter(); model=load_mujoco_model(preset_id=c["preset_id"],model_type="mjcf"); data=mujoco.MjData(model); robot=FeagineMjcfRobot(model,data,preset_id=c["preset_id"]); robot.drive_section_angles(c["protocol"]["section_angles"]); robot.set_grip_command(c["protocol"]["gripper_command"]); robot.drive_grasper_rotation(c["protocol"]["grasper_rotation_rad"])
 tracked=[]
 for j in range(model.njnt):
  if model.jnt_limited[j]:
   name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j); adr=int(model.jnt_qposadr[j]); lo,hi=map(float,model.jnt_range[j]); typ=int(model.jnt_type[j]); tracked.append({"name":name,"qpos_adr":adr,"joint_type":typ,"lower":lo,"upper":hi,"min":float(data.qpos[adr]),"max":float(data.qpos[adr]),"max_penetration":0.0})
 finite=True
 for _ in range(c["protocol"]["steps"]):
  mujoco.mj_step(model,data); finite=finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.ctrl).all())
  for x in tracked:
   q=float(data.qpos[x["qpos_adr"]]); x["min"]=min(x["min"],q); x["max"]=max(x["max"],q); x["max_penetration"]=max(x["max_penetration"],x["lower"]-q,q-x["upper"],0.0)
 violating=[x for x in tracked if x["max_penetration"]>1e-12]; hinge=[x for x in violating if x["joint_type"]==mujoco.mjtJoint.mjJNT_HINGE]; slide=[x for x in violating if x["joint_type"]==mujoco.mjtJoint.mjJNT_SLIDE]; max_h=max([x["max_penetration"] for x in hinge],default=0.0); max_s=max([x["max_penetration"] for x in slide],default=0.0); material=max_h>c["diagnostic_thresholds"]["hinge_material_violation_rad"] or max_s>c["diagnostic_thresholds"]["slide_material_violation_m"]; decision="MATERIAL_LIMIT_VIOLATION" if material else "SOFT_LIMIT_PENETRATION_WITHIN_DIAGNOSTIC_THRESHOLD"
 result={"classification":"PASS" if finite and abs(float(data.time)-1.0)<1e-12 else "FAIL","experiment":c["experiment_id"],"decision":decision,"execution_pass":finite,"steps":c["protocol"]["steps"],"simulated_time_s":float(data.time),"limited_joints_checked":len(tracked),"violating_joint_count":len(violating),"max_hinge_penetration_rad":max_h,"max_slide_penetration_m":max_s,"violations":violating,"diagnostic_thresholds":c["diagnostic_thresholds"],"elapsed_s":time.perf_counter()-start,"cuda_initialized":False,"model_predictions":0,"reward":None,"success":None,"hardware_commands":0,"source_hashes":{"config":sha(C)},"boundary":c["boundary"]}; out.mkdir(parents=True); (out/"result.json").write_text(json.dumps(result,indent=2)+"\n"); rp.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["classification"]=="PASS" else 1)
if __name__=="__main__": main()
