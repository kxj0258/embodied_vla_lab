#!/usr/bin/env python3
"""CPU-only exact-copy initialization and backward gate for Phase 5D."""
import hashlib,json,random,sys,time
from pathlib import Path
import h5py,numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts")); from phase5_residual_world_model import ResidualWorldModel,count_parameters
CONFIG=ROOT/"configs/train/phase5d_residual_world_model_static_v1.yaml"; RESULT=ROOT/"results/phase5d_residual_world_model_static_v1.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 started=time.perf_counter(); c=yaml.safe_load(CONFIG.read_text()); random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"]); torch.set_num_threads(2); assert not torch.cuda.is_initialized()
 original=set(yaml.safe_load((ROOT/"configs/dataset/phase5a_libero_world_model_contract_v1.yaml").read_text())["split"]["train_episodes"]); train=set(c["split"]["train_episodes"]); dev=set(c["split"]["development_episodes"]); confirm=set(c["split"]["prior_exposed_confirmation_episodes"])
 with h5py.File(ROOT/c["dataset"]["path"],"r") as f:
  g=f["data/demo_2"]; images=torch.from_numpy(g[c["dataset"]["image_source"]][:5].copy()).permute(0,3,1,2).float()/255; actions=torch.from_numpy(g[c["dataset"]["action_source"]][:4].copy()).float()
 x=images[:1].repeat(2,1,1,1); a=actions[None].repeat(2,1,1); y=images[1:5][None].repeat(2,1,1,1,1); models={"residual_action_free":ResidualWorldModel(False),"residual_action_conditioned":ResidualWorldModel(True)}; details={}
 for name,m in models.items():
  pred=m.rollout(x,a); initial_max=float((pred-x[:,None]).abs().max()); loss=(pred-y).abs().mean(); loss.backward(); grads=[p.grad for p in m.parameters() if p.grad is not None]; details[name]={"parameters":count_parameters(m),"rollout_shape":list(pred.shape),"initial_copy_max_abs":initial_max,"initial_pixel_l1_0_1":float(loss.detach()),"gradient_tensors":len(grads),"gradients_finite":all(torch.isfinite(q).all().item() for q in grads),"nonzero_gradient_tensors":sum(int(q.abs().max()>0) for q in grads)}
 conditioned=models["residual_action_conditioned"].eval()
 with torch.no_grad(): latent_effect=float((conditioned.latent_rollout(x[:1],a[:1])-conditioned.latent_rollout(x[:1],torch.zeros_like(a[:1]))).abs().mean())
 counts=[d["parameters"] for d in details.values()]; checks={"phase5c_no_go_frozen":yaml.safe_load((ROOT/"results/phase5c_world_model_train100_pair_v1.yaml").read_text())["decision"]=="ENGINEERING_PASS_OFFLINE_BASELINE_NO_GO","nested_train_dev_exact":train|dev==original and not train&dev and len(train)==32 and len(dev)==8,"confirmation_disjoint":not (confirm&train) and not (confirm&dev) and c["split"]["confirmation_is_untouched_final"] is False,"development_source_real":2 in dev,"exact_copy_initialization":all(d["initial_copy_max_abs"]==0 for d in details.values()),"rollout_shapes":all(d["rollout_shape"]==[2,4,3,128,128] for d in details.values()),"finite_backward":all(d["gradients_finite"] and d["nonzero_gradient_tensors"]>0 for d in details.values()),"latent_action_path_effective":latent_effect>0,"small_models":max(counts)<2_000_000,"parameter_matched":max(counts)/min(counts)<1.05,"cpu_only":not torch.cuda.is_initialized(),"training_not_authorized":c["future_development_gate"]["formal_training_authorized"] is False and c["future_development_gate"]["confirmation_execution_authorized"] is False}
 result={"classification":"PASS" if all(checks.values()) else "FAIL","experiment":c["experiment_id"],"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"split":{"train":sorted(train),"development":sorted(dev),"prior_exposed_confirmation":sorted(confirm)},"models":details,"conditioned_latent_action_effect_mean_abs":latent_effect,"cpu_wall_s":time.perf_counter()-started,"cuda_initialized":False,"training_steps":0,"optimizer_steps":0,"environment_steps":0,"source_hashes":{"config":sha(CONFIG),"implementation":sha(ROOT/"scripts/phase5_residual_world_model.py")},"boundary":c["boundary"]}
 if RESULT.exists(): raise FileExistsError(f"refusing overwrite: {RESULT}")
 RESULT.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["classification"]=="PASS" else 1)
if __name__=="__main__": main()
