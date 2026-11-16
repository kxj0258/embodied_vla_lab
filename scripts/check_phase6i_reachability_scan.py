#!/usr/bin/env python3
import hashlib,math,sys,time
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]; C=ROOT/"configs/task/phase6i_feagine_a03_reachability_scan_v1.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def wrap(v): return (v+math.pi)%(2*math.pi)-math.pi
def canonical(values):
 out=[]
 for theta,phi in zip(values[0::2],values[1::2]):
  if theta<0: theta=-theta; phi=wrap(phi+math.pi)
  else: phi=wrap(phi)
  out.extend([theta,phi])
 return np.asarray(out,dtype=np.float64)
def make_pose(core):
 p=core.Pose(); p.position=np.zeros(3,dtype=np.float32); p.quaternion=np.array([1,0,0,0],dtype=np.float32); return p
def sections(core,v): return [core.PccSectionConfig(.16,float(v[i]),float(v[i+1])) for i in range(0,6,2)]
def direction(p):
 w,x,y,z=np.asarray(p.quaternion,dtype=np.float64); r=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]); return r@np.array([0.,0.,1.])
def main():
 c=yaml.safe_load(C.read_text()); rp=ROOT/c["structured_result"]
 if rp.exists(): raise FileExistsError(f"refusing overwrite: {rp}")
 import pyfeagine_sim_core as core
 start=time.perf_counter(); base=make_pose(core); straight=core.compute_forward_kinematics(base,sections(core,[0,0,0,0,0,0]),5).section_end_poses[-1]; origin=np.asarray(straight.position,dtype=np.float64); axis=direction(straight); rows=[]
 for spec in c["lateral_targets"]:
  scan=[]
  for retract in c["axial_retraction_scan_m"]:
   target=core.IkTarget(); target.position=(origin+np.array([spec["delta_xy_m"][0],spec["delta_xy_m"][1],-retract])).astype(np.float32); target.tip_direction=axis.astype(np.float32); attempts=[]
   for seed in c["multistart_seeds"]:
    opts=core.InverseKinematicsOptions(); opts.max_iterations=c["ik"]["max_iterations"]; opts.position_tolerance=c["ik"]["position_tolerance_m"]; opts.direction_tolerance=c["ik"]["direction_tolerance"]; opts.clamp_theta=True; opts.theta_limit=c["ik"]["theta_limit_rad"]
    solved=core.compute_inverse_kinematics(base,sections(core,seed["angles_rad"]),target,opts); raw=np.asarray([v for s in solved.solved_sections for v in (float(s.theta),float(s.phi))]); can=canonical(raw); fk=core.compute_forward_kinematics(base,sections(core,can),5).section_end_poses[-1]; pe=float(np.linalg.norm(np.asarray(fk.position)-np.asarray(target.position))); de=float(np.linalg.norm(direction(fk)-axis)); valid=bool(solved.success and pe<=c["ik"]["position_tolerance_m"] and de<=c["ik"]["direction_tolerance"]); attempts.append({"seed_id":seed["id"],"solver_success":bool(solved.success),"position_error_m":pe,"direction_error":de,"canonical_angles_rad":can.tolist(),"valid":valid})
   valid=[x for x in attempts if x["valid"]]; selected=min(valid,key=lambda x:(x["position_error_m"],x["direction_error"],x["seed_id"])) if valid else None; scan.append({"axial_retraction_m":retract,"reachable":selected is not None,"selected":selected,"best_position_error_m":min(x["position_error_m"] for x in attempts)})
  reachable=[x for x in scan if x["reachable"]]; rows.append({"target_id":spec["id"],"minimum_retraction_m":reachable[0]["axial_retraction_m"] if reachable else None,"first_reachable":reachable[0] if reachable else None,"scan":scan})
 mins=[x["minimum_retraction_m"] for x in rows if x["minimum_retraction_m"] is not None]; spread=max(mins)-min(mins) if mins else None; checks={"frozen_grid":c["axial_retraction_scan_m"]==[i*.00025 for i in range(21)],"all_directions_reachable":len(mins)==c["gate"]["required_directions_reachable"],"symmetric_threshold":spread is not None and spread<=c["gate"]["maximum_minimum_retraction_spread_m"],"source_phase6f_immutable":sha(ROOT/"results/phase6f_feagine_a03_external_targets_dev4_v1.yaml")=="3373d5fe82ae756e73e7313e1c777bc130e1a9c410e69993692e30063a261c20","no_mujoco_import":"mujoco" not in sys.modules,"cpu_only":not any(x.startswith("torch") for x in sys.modules)}; result={"classification":"PASS" if all(checks.values()) else "FAIL","experiment":c["experiment_id"],"decision":"REACHABILITY_SCAN_PASS" if all(checks.values()) else "REACHABILITY_SCAN_FAIL","checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"directions":rows,"minimum_retractions_m":{x["target_id"]:x["minimum_retraction_m"] for x in rows},"minimum_retraction_spread_m":spread,"total_ik_attempts":len(c["lateral_targets"])*len(c["axial_retraction_scan_m"])*len(c["multistart_seeds"]),"elapsed_s":time.perf_counter()-start,"environment_steps":0,"cuda_initialized":False,"model_predictions":0,"hardware_commands":0,"source_hashes":{"config":sha(C),"phase6f_result":sha(ROOT/"results/phase6f_feagine_a03_external_targets_dev4_v1.yaml"),"phase6h_result":sha(ROOT/"results/phase6h_feagine_signed_theta_multistart_static_v1.yaml")},"boundary":c["boundary"]}; rp.write_text(yaml.safe_dump(result,sort_keys=False)); print(yaml.safe_dump(result,sort_keys=False)); raise SystemExit(0 if result["classification"]=="PASS" else 1)
if __name__=="__main__": main()
