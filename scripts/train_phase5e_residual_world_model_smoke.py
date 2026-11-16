#!/usr/bin/env python3
"""Paired residual world-model development smoke; host-authorized only."""
import hashlib,json,math,random,sys,time
from pathlib import Path
import h5py,numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts")); from phase5_residual_world_model import ResidualWorldModel,count_parameters; from train_phase5c_world_model_smoke import load_windows,tensors,evaluate,copy_metrics
CONFIG=ROOT/"configs/train/phase5e_residual_world_model_dev_smoke_v1.yaml"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for x in iter(lambda:f.read(8*1024*1024),b""): h.update(x)
 return h.hexdigest()
def main():
 c=yaml.safe_load(CONFIG.read_text()); out=ROOT/c["output_dir"]
 for p in [out,ROOT/c["structured_result"]]:
  if p.exists(): raise FileExistsError(f"refusing overwrite: {p}")
 mp=ROOT/c["fixture"]["manifest"]; m=json.loads(mp.read_text()); assert sha(mp)=="6a5da23a0c1e3179ba24749eaadb9f91a1e34df6187275f44aedd69088a0d6b4" and not m["prior_exposed_confirmation_included"]
 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 if device.type!="cuda": raise RuntimeError("Phase5E host smoke requires CUDA")
 out.mkdir(parents=True); (out/"resolved_config.yaml").write_text(yaml.safe_dump(c,sort_keys=False)); random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"]); torch.cuda.manual_seed_all(c["seed"]); torch.cuda.reset_peak_memory_stats()
 with h5py.File(ROOT/c["dataset"]["path"],"r") as f:
  train=load_windows(f,m["train_windows"],c["dataset"]["image_source"],c["dataset"]["action_source"]); dev=load_windows(f,m["development_windows"],c["dataset"]["image_source"],c["dataset"]["action_source"])
 rng=np.random.default_rng(c["seed"]); batches=[]
 while len(batches)<c["training"]["updates_per_model"]:
  p=rng.permutation(len(train[0])); batches.extend([p[i:i+c["training"]["batch_size"]].tolist() for i in range(0,len(p),c["training"]["batch_size"])])
 batches=batches[:c["training"]["updates_per_model"]]; torch.manual_seed(c["seed"]); free=ResidualWorldModel(False); torch.manual_seed(c["seed"]); cond=ResidualWorldModel(True); shared=all(torch.equal(free.state_dict()[k],cond.state_dict()[k]) for k in free.state_dict() if k.startswith("encoder.") or k.startswith("decoder.")); assert shared
 results={}; copy=copy_metrics(dev); started=time.perf_counter()
 for name,model in [("residual_action_free",free),("residual_action_conditioned",cond)]:
  model=model.to(device); pre=evaluate(model,dev,device); opt=torch.optim.AdamW(model.parameters(),lr=c["training"]["learning_rate"],weight_decay=c["training"]["weight_decay"]); hist=[]; ts=time.perf_counter()
  for step,idx in enumerate(batches,1):
   model.train(); x,a,y=tensors(train,np.asarray(idx),device); opt.zero_grad(set_to_none=True); pred=model.rollout(x,a); loss=(pred-y).abs().mean(); loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),c["training"]["max_grad_norm"])); opt.step(); hist.append({"step":step,"loss":float(loss.detach()),"grad_norm_preclip":grad})
  post=evaluate(model,dev,device); ck=out/f"{name}_step100.pt"; torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"step":100,"config":c},ck); results[name]={"parameters":count_parameters(model),"pre_development":pre,"post_development":post,"history":hist,"train_plus_postdev_s":time.perf_counter()-ts,"checkpoint":{"path":ck.relative_to(ROOT).as_posix(),"bytes":ck.stat().st_size,"sha256":sha(ck)}}; del model,opt; torch.cuda.empty_cache()
 finite=all(math.isfinite(row[k]) for v in results.values() for row in v["history"] for k in ["loss","grad_norm_preclip"]); h4c=results["residual_action_conditioned"]["post_development"][4]["pixel_mae_0_255"]; h4f=results["residual_action_free"]["post_development"][4]["pixel_mae_0_255"]; h4copy=copy[4]["pixel_mae_0_255"]; decision="DEVELOPMENT_GO_ACTION_CONDITIONING" if h4c<h4f and h4c<h4copy else "DEVELOPMENT_NO_GO_ACTION_CONDITIONING"
 result={"classification":"PASS" if finite else "FAIL","experiment":c["experiment_id"],"execution_pass":finite,"decision":decision,"device":str(device),"fixture_sha256":sha(mp),"shared_encoder_decoder_initialization":shared,"identical_batch_order":True,"copy_frame_development":copy,"models":results,"elapsed_s":time.perf_counter()-started,"peak_vram_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_vram_reserved_bytes":torch.cuda.max_memory_reserved(),"training_steps_total":200,"optimizer_steps_total":200,"prior_exposed_confirmation_executed":False,"checkpoint_selected":False,"environment_steps":0,"boundary":c["boundary"]}; rp=out/"result.json"; rp.write_text(json.dumps(result,indent=2)+"\n"); (ROOT/c["structured_result"]).write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
