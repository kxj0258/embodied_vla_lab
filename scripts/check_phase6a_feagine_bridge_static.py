#!/usr/bin/env python3
"""CPU-only numerical and safety contract gate for Phase 6A."""
import hashlib,json,random,sys,time
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
from feagine_bridge import ContractError,PCCState,adapt_libero_delta,pcc_forward,pcc_inverse_position
CONFIG=ROOT/"configs/task/phase6a_feagine_bridge_static_v4.yaml"; RESULT=ROOT/"results/phase6a_feagine_bridge_static_v4.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rejected(fn):
 try: fn(); return False
 except ContractError: return True
def main():
 started=time.perf_counter(); c=yaml.safe_load(CONFIG.read_text()); random.seed(c["seed"]); rng=np.random.default_rng(c["seed"]); limits=c["pcc_limits"]; errors=[]
 for _ in range(c["validation"]["random_roundtrips"]):
  length=float(rng.uniform(limits["min_length_m"]+.005,limits["max_length_m"]-.005)); max_theta=min(limits["max_bend_angle_rad"]-.05,limits["max_abs_curvature_per_m"]*length-.05); theta=float(rng.uniform(0,max_theta)); phi=float(rng.uniform(-np.pi,np.pi)); q=PCCState(theta,phi,length); xyz=pcc_forward(q); q2=pcc_inverse_position(xyz,limits); errors.append(float(np.linalg.norm(pcc_forward(q2)-xyz)))
 straight=pcc_inverse_position([0,0,0.2],limits); nominal=adapt_libero_delta([.001,-.002,0,0,0,0,.5],[.02,.01,.2],c); clipped=adapt_libero_delta([.1,0,0,0,0,0,2.0],[0,0,.2],c)
 checks={
  "hardware_parameters_explicitly_missing":not c["hardware_binding"]["calibrated_parameters_available"] and not c["hardware_binding"]["mjcf_available"] and not c["hardware_binding"]["actuator_mapping_available"],
  "no_hardware_fidelity_claim":c["hardware_binding"]["claim_real_robot_fidelity"] is False,
  "roundtrip_count":len(errors)==256,
  "roundtrip_tolerance":max(errors)<=c["validation"]["position_tolerance_m"],
  "straight_case":straight==PCCState(0.0,0.0,0.2),
  "nominal_adapter_finite":np.isfinite(nominal["target_tip_continuum_m"]).all(),
  "translation_rate_limit":bool(clipped["translation_clipped"] and np.isclose(np.linalg.norm(clipped["target_tip_continuum_m"]-np.array([0,0,.2])),.01)),
  "gripper_limit":clipped["gripper"]==1.0,
  "reject_nan":rejected(lambda:adapt_libero_delta([np.nan,0,0,0,0,0,0],[0,0,.2],c)),
  "reject_rotation":rejected(lambda:adapt_libero_delta([0,0,0,.01,0,0,0],[0,0,.2],c)),
  "reject_workspace":rejected(lambda:adapt_libero_delta([0,0,0,0,0,0,0],[0,0,.31],c)),
  "reject_unreachable_pcc":rejected(lambda:pcc_inverse_position([.3,0,.01],limits)),
  "all_execution_forbidden":not any(c["authorization"].values()),
  "cpu_only":"torch" not in sys.modules,
 }
 checks={key:bool(value) for key,value in checks.items()}
 result={"classification":"PASS" if all(checks.values()) else "FAIL","experiment":c["experiment_id"],"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"roundtrip":{"samples":len(errors),"max_position_error_m":max(errors),"mean_position_error_m":float(np.mean(errors))},"nominal_adapter":{"target_tip_continuum_m":nominal["target_tip_continuum_m"].tolist(),"pcc_state":nominal["pcc_state"].__dict__,"gripper":nominal["gripper"]},"source_hashes":{"config":sha(CONFIG),"implementation":sha(ROOT/"scripts/feagine_bridge.py")},"cpu_wall_s":time.perf_counter()-started,"cuda_initialized":False,"model_predictions":0,"environment_steps":0,"hardware_commands":0,"boundary":c["boundary"]}
 if RESULT.exists(): raise FileExistsError(f"refusing overwrite: {RESULT}")
 RESULT.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["classification"]=="PASS" else 1)
if __name__=="__main__": main()
