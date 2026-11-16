#!/usr/bin/env python3
"""Run one presentation-only HD success recapture for each frozen breadth task."""
from pathlib import Path
import hashlib, json, os, subprocess, tempfile, yaml
ROOT=Path(__file__).resolve().parents[1]; CHILD=ROOT/"scripts/record_p0_hd_openvla_libero.py"; RESULT=ROOT/"results/p0_breadth_hd_10task_v1_manifest.yaml"
TASKS=[
(0,"open_the_middle_drawer_of_the_cabinet","open the middle drawer of the cabinet",0,20261001,114),
(1,"put_the_bowl_on_the_stove","put the bowl on the stove",1,20261005,90),
(2,"put_the_wine_bottle_on_top_of_the_cabinet","put the wine bottle on top of the cabinet",0,20261007,77),
(3,"open_the_top_drawer_and_put_the_bowl_inside","open the top drawer and put the bowl inside",2,20261012,246),
(4,"put_the_bowl_on_top_of_the_cabinet","put the bowl on top of the cabinet",0,20261013,93),
(5,"push_the_plate_to_the_front_of_the_stove","push the plate to the front of the stove",0,20261016,133),
(6,"put_the_cream_cheese_in_the_bowl","put the cream cheese in the bowl",1,20261020,108),
(7,"turn_on_the_stove","turn on the stove",0,20261022,89),
(8,"put_the_bowl_on_the_plate","put the bowl on the plate",0,20261025,76),
(9,"put_the_wine_bottle_on_the_rack","put the wine bottle on the rack",0,20261028,270)]
def sha(p): h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
 if RESULT.exists(): raise FileExistsError(f"refusing overwrite: {RESULT}")
 records=[]
 for tid,name,instruction,init,seed,expected_steps in TASKS:
  eid=f"p0_breadth_hd_base_task{tid:02d}_init{init:02d}_v1_seed{seed}"; out=ROOT/f"outputs/presentation/{eid}"; result=ROOT/f"results/{eid}.yaml"
  if out.exists() or result.exists(): raise FileExistsError(f"refusing overwrite: {out} or {result}")
  cfg={"experiment_id":eid,"kind":"presentation_only_breadth_hd_capture_single_egl","task":{"suite":"libero_goal","task_id":tid,"name":name,"instruction":instruction},"model":{"python":"/home/kxj/miniforge3/envs/evla-openvla/bin/python","worker":"scripts/run_phase3a_openvla_persistent_worker.py","revision":"fa5ae1e7509348889295bba8e08621d8b55e9baf","policy":"frozen_base"},"protocol":{"schema_version":"evla.openvla.single_step.v1","evaluator_seed":seed,"init_state_index":init,"render_resolution":[640,640],"model_ipc_resolution":[128,128],"model_image_preprocess":"rotate_180_then_openvla_center_crop_90_percent","display_render":"same_sim_same_egl_direct_640_camera_render","control_frequency_hz":20},"presentation_label":f"OpenVLA-7B · frozen base · Task {tid}","output_dir":out.relative_to(ROOT).as_posix(),"structured_result":result.relative_to(ROOT).as_posix(),"raw_log":"captured_by_parent_host_log","boundary":"Presentation-only recapture of the first successful state selected deterministically from the frozen 10-task/30-episode result; cannot alter 20/30."}
  with tempfile.NamedTemporaryFile("w",suffix=".yaml",delete=False,dir="/tmp") as f: yaml.safe_dump(cfg,f,sort_keys=False); cp=f.name
  try: subprocess.run(["/home/kxj/miniforge3/envs/evla-core/bin/python",str(CHILD)],cwd=ROOT,env={**os.environ,"EVLA_P0_HD_CONFIG":cp},check=True)
  finally: Path(cp).unlink(missing_ok=True)
  data=yaml.safe_load(result.read_text()); assert data["policy_success"] and data["steps"]==expected_steps
  records.append({"task_id":tid,"task_name":name,"instruction":instruction,"selected_init_state":init,"source_eval_seed":seed,"expected_and_observed_steps":expected_steps,"result":result.relative_to(ROOT).as_posix(),"result_sha256":sha(result),"presentation_video":data["artifacts"]["presentation_video"],"clean_video":data["artifacts"]["clean_video"]})
 manifest={"manifest":"p0_breadth_hd_10task_v1","classification":"PASS","selection_rule":"first successful init state in frozen task order; fixed before recapture","source_result":"results/p0_breadth_openvla_base_10task30_v2.yaml","source_success":"20/30","task_coverage":"10/10","captures":records,"boundary":"Presentation-only; no failures rerun or aggregate metrics changed."}; RESULT.write_text(yaml.safe_dump(manifest,sort_keys=False)); print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
