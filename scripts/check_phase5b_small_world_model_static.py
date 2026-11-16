#!/usr/bin/env python3
"""CPU-only real-data forward/backward smoke for Phase 5B."""
import hashlib, json, random, sys, time
from pathlib import Path
import h5py, numpy as np, torch, yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts")); from phase5_small_world_model import SmallWorldModel,count_parameters
CONFIG=ROOT/"configs/train/phase5b_small_world_model_static_v1.yaml"; RESULT=ROOT/"results/phase5b_small_world_model_static_v1.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    started=time.perf_counter(); c=yaml.safe_load(CONFIG.read_text()); random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"]); torch.set_num_threads(2)
    assert not torch.cuda.is_initialized(); path=ROOT/c["dataset"]["path"]; assert sha(path)==c["dataset"]["sha256"]
    with h5py.File(path,"r") as f:
        g=f["data/demo_0"]; images=torch.from_numpy(g[c["dataset"]["image_source"]][:9].copy()).permute(0,3,1,2).float()/255; actions=torch.from_numpy(g[c["dataset"]["action_source"]][:8].copy()).float()
    image0=images[:1].repeat(2,1,1,1); action_seq=actions.unsqueeze(0).repeat(2,1,1); targets=images[1:9].unsqueeze(0).repeat(2,1,1,1,1)
    models={"action_free":SmallWorldModel(False),"action_conditioned":SmallWorldModel(True)}; outputs={}; grads={}; losses={}
    for name,m in models.items():
        pred=m.rollout(image0,action_seq); recon=m.reconstruct(image0); loss=(pred-targets).abs().mean()+(recon-image0).abs().mean(); loss.backward(); outputs[name]=pred.detach(); losses[name]=float(loss); vals=[p.grad for p in m.parameters() if p.grad is not None]; grads[name]={"tensors":len(vals),"all_finite":all(torch.isfinite(x).all().item() for x in vals),"global_l2":float(torch.sqrt(sum((x*x).sum() for x in vals)))}
    conditioned=models["action_conditioned"].eval(); zero=torch.zeros_like(action_seq)
    with torch.no_grad(): action_effect=float((conditioned.rollout(image0[:1],action_seq[:1])-conditioned.rollout(image0[:1],zero[:1])).abs().mean())
    counts={name:count_parameters(m) for name,m in models.items()}; ratio=counts["action_conditioned"]/counts["action_free"]
    checks={"data_contract_pass":yaml.safe_load((ROOT/c["data_contract"]).read_text())["classification"]=="PASS","dataset_hash":True,"cpu_only":not torch.cuda.is_initialized(),"real_image_shape":list(image0.shape)==[2,3,128,128],"real_action_shape":list(action_seq.shape)==[2,8,7],"free_rollout_shape":list(outputs["action_free"].shape)==[2,8,3,128,128],"conditioned_rollout_shape":list(outputs["action_conditioned"].shape)==[2,8,3,128,128],"outputs_finite":all(torch.isfinite(x).all().item() for x in outputs.values()),"outputs_in_0_1":all((x.min()>=0 and x.max()<=1) for x in outputs.values()),"losses_finite":all(np.isfinite(x) for x in losses.values()),"gradients_finite":all(x["all_finite"] for x in grads.values()),"action_path_effective":action_effect>0,"parameter_scale_small":max(counts.values())<2_000_000,"parameter_counts_comparable":ratio<1.05,"formal_training_not_authorized":c["training_candidate"]["formal_training_authorized"] is False,"fresh_target":not (ROOT/c["output_dir"]).exists()}
    result={"classification":"PASS" if all(checks.values()) else "FAIL","experiment":c["experiment_id"],"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"real_data":{"source_episode":0,"source_frames":[0,8],"image_batch_shape":list(image0.shape),"action_sequence_shape":list(action_seq.shape),"target_shape":list(targets.shape)},"parameters":{**counts,"conditioned_to_free_ratio":ratio},"random_weight_smoke":{"losses":losses,"gradients":grads,"action_effect_mean_abs":action_effect},"timing":{"cpu_wall_s":time.perf_counter()-started},"cuda_initialized":torch.cuda.is_initialized(),"training_steps":0,"optimizer_steps":0,"environment_resets":0,"environment_steps":0,"model_predictions":0,"source_hashes":{"config":sha(CONFIG),"model_implementation":sha(ROOT/"scripts/phase5_small_world_model.py")},"boundary":c["boundary"]}
    if RESULT.exists(): raise FileExistsError(f"refusing overwrite: {RESULT}")
    RESULT.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["classification"]=="PASS" else 1)
if __name__=="__main__": main()
