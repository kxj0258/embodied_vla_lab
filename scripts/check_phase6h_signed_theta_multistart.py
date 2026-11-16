#!/usr/bin/env python3
import hashlib,math,sys,time
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]; C=ROOT/"configs/task/phase6h_feagine_signed_theta_multistart_static_v1.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def wrap(v): return (v+math.pi)%(2*math.pi)-math.pi
def canonical(values):
 out=[]
 for theta,phi in zip(values[0::2],values[1::2]):
  if theta<0: theta=-theta; phi=wrap(phi+math.pi)
  else: phi=wrap(phi)
  out.extend([theta,phi])
 return np.asarray(out,dtype=np.float64)
def pose():
 import pyfeagine_sim_core as core
 p=core.Pose(); p.position=np.zeros(3,dtype=np.float32); p.quaternion=np.array([1,0,0,0],dtype=np.float32); return p
def sections(core,values): return [core.PccSectionConfig(.16,float(values[i]),float(values[i+1])) for i in range(0,6,2)]
def direction(p):
 w,x,y,z=np.asarray(p.quaternion,dtype=np.float64); r=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]); return r@np.array([0.,0.,1.])
def main():
 c=yaml.safe_load(C.read_text()); rp=ROOT/c["structured_result"]
 if rp.exists(): raise FileExistsError(f"refusing overwrite: {rp}")
 import pyfeagine_sim_core as core
 start=time.perf_counter(); base=pose(); rng=np.random.default_rng(c["equivalence_test"]["seed"]); max_pos=max_dir=0.0
 samples=[rng.uniform([-1.0,-math.pi]*3,[1.0,math.pi]*3) for _ in range(c["equivalence_test"]["random_samples"])]
 phase6g=yaml.safe_load((ROOT/"results/phase6g_feagine_a03_mapping_diagnostic_v1.yaml").read_text()); samples += [np.asarray(x["raw_solved_section_angles_rad"],dtype=np.float64) for x in phase6g["episodes"]]
 for raw in samples:
  can=canonical(raw); a=core.compute_forward_kinematics(base,sections(core,raw),5).section_end_poses[-1]; b=core.compute_forward_kinematics(base,sections(core,can),5).section_end_poses[-1]; max_pos=max(max_pos,float(np.linalg.norm(np.asarray(a.position)-np.asarray(b.position)))); max_dir=max(max_dir,float(np.linalg.norm(direction(a)-direction(b))))
 straight=core.compute_forward_kinematics(base,sections(core,[0,0,0,0,0,0]),5).section_end_poses[-1]; straight_pos=np.asarray(straight.position,dtype=np.float64); straight_dir=direction(straight); rows=[]
 for target_spec in c["targets"]:
  target=core.IkTarget(); target.position=(straight_pos+np.asarray(target_spec["delta_m"],dtype=np.float64)).astype(np.float32); target.tip_direction=straight_dir.astype(np.float32); attempts=[]
  for seed in c["multistart_seeds"]:
   opts=core.InverseKinematicsOptions(); opts.max_iterations=c["ik"]["max_iterations"]; opts.position_tolerance=c["ik"]["position_tolerance_m"]; opts.direction_tolerance=c["ik"]["direction_tolerance"]; opts.clamp_theta=True; opts.theta_limit=c["ik"]["theta_limit_rad"]
   result=core.compute_inverse_kinematics(base,sections(core,seed["angles_rad"]),target,opts); raw=np.asarray([v for s in result.solved_sections for v in (float(s.theta),float(s.phi))]); can=canonical(raw); fk=core.compute_forward_kinematics(base,sections(core,can),5).section_end_poses[-1]; pos_err=float(np.linalg.norm(np.asarray(fk.position)-np.asarray(target.position))); dir_err=float(np.linalg.norm(direction(fk)-np.asarray(target.tip_direction))); valid=bool(result.success and pos_err<=c["ik"]["position_tolerance_m"] and dir_err<=c["ik"]["direction_tolerance"] and np.all(can[0::2]>=c["gate"]["canonicalized_theta_min_rad"]) and np.all(can[0::2]<=c["gate"]["canonicalized_theta_max_rad"])); attempts.append({"seed_id":seed["id"],"solver_success":bool(result.success),"iterations":int(result.iterations),"raw_angles_rad":raw.tolist(),"canonical_angles_rad":can.tolist(),"canonical_position_error_m":pos_err,"canonical_direction_error":dir_err,"valid":valid})
  valid=[x for x in attempts if x["valid"]]; selected=min(valid,key=lambda x:(x["canonical_position_error_m"],x["canonical_direction_error"],x["seed_id"])) if valid else None; rows.append({"target_id":target_spec["id"],"valid_seed_count":len(valid),"selected":selected,"attempts":attempts})
 checks={"fk_position_equivalence":max_pos<=c["equivalence_test"]["position_tolerance_m"],"fk_direction_equivalence":max_dir<=c["equivalence_test"]["direction_tolerance"],"all_targets_solved":sum(x["selected"] is not None for x in rows)==c["gate"]["required_target_solutions"],"no_mujoco_import":"mujoco" not in sys.modules,"cpu_only":not any(x.startswith("torch") for x in sys.modules)}; result={"classification":"PASS" if all(checks.values()) else "FAIL","experiment":c["experiment_id"],"decision":"SIGNED_THETA_AND_MULTISTART_STATIC_PASS" if all(checks.values()) else "SIGNED_THETA_AND_MULTISTART_STATIC_FAIL","checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"equivalence_samples":len(samples),"max_fk_position_difference_m":max_pos,"max_fk_direction_difference":max_dir,"target_solutions":rows,"elapsed_s":time.perf_counter()-start,"environment_steps":0,"cuda_initialized":False,"model_predictions":0,"hardware_commands":0,"source_hashes":{"config":sha(C),"phase6g_result":sha(ROOT/"results/phase6g_feagine_a03_mapping_diagnostic_v1.yaml")},"boundary":c["boundary"]}; rp.write_text(yaml.safe_dump(result,sort_keys=False)); print(yaml.safe_dump(result,sort_keys=False)); raise SystemExit(0 if result["classification"]=="PASS" else 1)
if __name__=="__main__": main()
