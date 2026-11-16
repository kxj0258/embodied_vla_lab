#!/usr/bin/env python3
"""Paired 100-update world-model smoke; requires explicit host authorization."""
import hashlib,json,math,os,random,sys,time
from pathlib import Path
import h5py,numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts")); from phase5_small_world_model import SmallWorldModel,count_parameters
CONFIG=ROOT/"configs/train/phase5c_world_model_train_smoke_v1.yaml"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for x in iter(lambda:f.read(8*1024*1024),b""): h.update(x)
 return h.hexdigest()
def load_windows(f,refs,image_key,action_key):
 xs=[]; acts=[]; ys=[]
 for r in refs:
  g=f[f"data/demo_{r['episode']}"]; t=r["start"]; h=r["horizon"]; xs.append(g[image_key][t]); acts.append(g[action_key][t:t+h]); ys.append(g[image_key][t+1:t+h+1])
 return np.stack(xs),np.stack(acts).astype("float32"),np.stack(ys)
def tensors(data,idx,device):
 x,a,y=data; return torch.from_numpy(x[idx]).permute(0,3,1,2).to(device=device,dtype=torch.float32).div_(255),torch.from_numpy(a[idx]).to(device),torch.from_numpy(y[idx]).permute(0,1,4,2,3).to(device=device,dtype=torch.float32).div_(255)
@torch.no_grad()
def evaluate(model,data,device,batch=8):
 model.eval(); sums={h:{"abs":0.,"sq":0.,"n":0} for h in range(1,5)}
 for s in range(0,len(data[0]),batch):
  idx=np.arange(s,min(s+batch,len(data[0]))); x,a,y=tensors(data,idx,device); p=model.rollout(x,a)
  for h in range(1,5):
   d=p[:,h-1]-y[:,h-1]; q=sums[h]; q["abs"]+=float(d.abs().sum()); q["sq"]+=float((d*d).sum()); q["n"]+=d.numel()
 return {h:{"pixel_mae_0_255":q["abs"]/q["n"]*255,"pixel_mse_0_255":q["sq"]/q["n"]*255*255,"psnr_db":10*math.log10(1/(q["sq"]/q["n"]))} for h,q in sums.items()}
@torch.no_grad()
def copy_metrics(data):
 x,_,y=data; out={}
 for h in range(1,5):
  d=x.astype("int16")-y[:,h-1].astype("int16"); mse=float((d.astype("float32")**2).mean()); out[h]={"pixel_mae_0_255":float(np.abs(d).mean()),"pixel_mse_0_255":mse,"psnr_db":10*math.log10(255*255/mse)}
 return out
def main():
 c=yaml.safe_load(CONFIG.read_text()); out=ROOT/c["output_dir"]; structured=ROOT/c["structured_result"]
 for p in [out,structured]:
  if p.exists(): raise FileExistsError(f"refusing overwrite: {p}")
 manifest_path=ROOT/c["fixture"]["manifest"]; m=json.loads(manifest_path.read_text()); assert sha(manifest_path)=="97b33ec833cdc6d5dfeae7288314e0436e8a15deebeeb789cc23cbc4d2336950" and sha(ROOT/c["dataset"]["path"])==c["dataset"]["sha256"]
 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
 if device.type!="cuda": raise RuntimeError("Phase5C host smoke requires CUDA")
 out.mkdir(parents=True); (out/"resolved_config.yaml").write_text(yaml.safe_dump(c,sort_keys=False)); random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"]); torch.cuda.manual_seed_all(c["seed"]); torch.cuda.reset_peak_memory_stats()
 with h5py.File(ROOT/c["dataset"]["path"],"r") as f:
  train=load_windows(f,m["train_windows"],c["dataset"]["image_source"],c["dataset"]["action_source"]); val=load_windows(f,m["validation_windows"],c["dataset"]["image_source"],c["dataset"]["action_source"])
 rng=np.random.default_rng(c["seed"]); batch_order=[]
 while len(batch_order)<c["training"]["updates_per_model"]:
  perm=rng.permutation(len(train[0])); batch_order.extend([perm[i:i+c["training"]["batch_size"]].tolist() for i in range(0,len(perm),c["training"]["batch_size"])])
 batch_order=batch_order[:c["training"]["updates_per_model"]]
 torch.manual_seed(c["seed"]); free=SmallWorldModel(False); torch.manual_seed(c["seed"]); conditioned=SmallWorldModel(True)
 shared_equal=all(torch.equal(free.state_dict()[k],conditioned.state_dict()[k]) for k in free.state_dict() if k.startswith("encoder.") or k.startswith("decoder.")); assert shared_equal
 copy=copy_metrics(val); results={}; started=time.perf_counter()
 for name,model in [("action_free",free),("action_conditioned",conditioned)]:
  model=model.to(device); pre=evaluate(model,val,device); opt=torch.optim.AdamW(model.parameters(),lr=c["training"]["learning_rate"],weight_decay=c["training"]["weight_decay"]); history=[]; train_start=time.perf_counter()
  for step,idx in enumerate(batch_order,1):
   model.train(); x,a,y=tensors(train,np.asarray(idx),device); opt.zero_grad(set_to_none=True); pred=model.rollout(x,a); recon=model.reconstruct(x); loss=(pred-y).abs().mean()+(recon-x).abs().mean(); loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),c["training"]["max_grad_norm"])); opt.step(); history.append({"step":step,"loss":float(loss.detach()),"grad_norm_preclip":grad})
  post=evaluate(model,val,device); ckpt=out/f"{name}_step100.pt"; torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"step":100,"config":c},ckpt); results[name]={"parameters":count_parameters(model),"pre_validation":pre,"post_validation":post,"history":history,"train_s":time.perf_counter()-train_start,"checkpoint":{"path":ckpt.relative_to(ROOT).as_posix(),"bytes":ckpt.stat().st_size,"sha256":sha(ckpt)}}; del model,opt; torch.cuda.empty_cache()
 finite=all(math.isfinite(row[k]) for v in results.values() for row in v["history"] for k in ["loss","grad_norm_preclip"]); result={"classification":"PASS" if finite else "FAIL","experiment":c["experiment_id"],"execution_pass":finite,"device":str(device),"fixture_sha256":sha(manifest_path),"shared_encoder_decoder_initialization":shared_equal,"identical_batch_order":True,"copy_frame_validation":copy,"models":results,"elapsed_s":time.perf_counter()-started,"peak_vram_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_vram_reserved_bytes":torch.cuda.max_memory_reserved(),"training_steps_total":200,"optimizer_steps_total":200,"environment_steps":0,"policy_predictions":0,"checkpoint_selected":False,"boundary":c["boundary"]}; rp=out/"result.json"; rp.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
