#!/usr/bin/env python3
"""CPU-only temporal contract and copy-frame baseline audit for Phase 5A."""
import hashlib, json, math
from pathlib import Path
import h5py, numpy as np, yaml
ROOT=Path(__file__).resolve().parents[1]; CONFIG=ROOT/"configs/dataset/phase5a_libero_world_model_contract_v1.yaml"; RESULT=ROOT/"results/phase5a_libero_world_model_data_contract_v1.yaml"
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def main():
    c=yaml.safe_load(CONFIG.read_text()); path=ROOT/c["dataset"]["path"]; assert digest(path)==c["dataset"]["sha256"]
    train=c["split"]["train_episodes"]; val=c["split"]["validation_episodes"]; hs=c["transition_contract"]["horizons"]
    assert len(train)==40 and len(val)==10 and not set(train)&set(val) and sorted(train+val)==list(range(50))
    splits={}; copy={view:{} for view in ["agentview","eye_in_hand"]}; all_actions=[]; action_deltas=[]; terminal_ok=True
    with h5py.File(path,"r") as f:
        assert int(f["data"].attrs["num_demos"])==50 and int(f["data"].attrs["total"])==4460
        for split,ids in [("train",train),("validation",val)]:
            lengths=[]; counts={h:0 for h in hs}; reward_sum=0; success_eps=0
            accum={v:{h:{"abs":0.0,"sq":0.0,"pixels":0} for h in hs} for v in copy}
            for e in ids:
                g=f[f"data/demo_{e}"]; n=len(g["actions"]); lengths.append(n); acts=g["actions"][()]; all_actions.append(acts)
                action_deltas.append(np.abs(np.diff(acts,axis=0))); assert acts.shape==(n,7) and g["robot_states"].shape==(n,9) and g["states"].shape==(n,79)
                done=g["dones"][()]; rew=g["rewards"][()]; terminal_ok=terminal_ok and bool(done[-1]) and not bool(done[:-1].any()); reward_sum+=int(rew.sum()); success_eps+=int(rew.max()>0)
                for view,key in [("agentview",c["dataset"]["primary_image"]),("eye_in_hand",c["dataset"]["secondary_image"])]:
                    x=g[key][()]; assert x.shape==(n,128,128,3) and x.dtype==np.uint8
                    for h in hs:
                        m=max(n-h,0); counts[h]+=m if view=="agentview" else 0
                        if not m: continue
                        d=x[h:].astype(np.int16)-x[:-h].astype(np.int16); a=accum[view][h]; a["abs"]+=float(np.abs(d).sum()); a["sq"]+=float((d.astype(np.float32)**2).sum()); a["pixels"]+=d.size
            splits[split]={"episodes":len(ids),"frames":sum(lengths),"episode_length_min":min(lengths),"episode_length_median":float(np.median(lengths)),"episode_length_max":max(lengths),"transition_samples_by_horizon":counts,"reward_sum":reward_sum,"successful_demonstrations":success_eps}
            if split=="validation":
                for view in copy:
                    for h in hs:
                        a=accum[view][h]; mae=a["abs"]/a["pixels"]; mse=a["sq"]/a["pixels"]; copy[view][h]={"samples":counts[h],"pixel_mae":mae,"pixel_mse":mse,"psnr_db":float(10*math.log10(255*255/mse)) if mse else None}
    actions=np.concatenate(all_actions); action_std=actions.std(0); action_delta=np.concatenate(action_deltas).mean(0)
    checks={"dataset_hash":True,"split_exact":True,"total_frames":splits["train"]["frames"]+splits["validation"]["frames"]==4460,"train_frames":splits["train"]["frames"]==3571,"validation_frames":splits["validation"]["frames"]==889,"all_horizons_nonempty":all(n>0 for s in splits.values() for n in s["transition_samples_by_horizon"].values()),"terminal_markers":terminal_ok,"actions_finite":bool(np.isfinite(actions).all()),"action_nonconstant":bool((action_std>1e-4).all()),"views_temporally_nonconstant":all(copy[v][1]["pixel_mae"]>0 for v in copy),"no_cross_episode_windows":True,"training_not_executed":True}
    result={"classification":"PASS" if all(checks.values()) else "FAIL","experiment_id":c["experiment_id"],"checks":checks,"splits":splits,"copy_current_frame_validation":copy,"actions":{"shape":[int(len(actions)),7],"min":actions.min(0).tolist(),"max":actions.max(0).tolist(),"std":action_std.tolist(),"mean_abs_consecutive_delta":action_delta.tolist(),"finite":True},"dataset_sha256":digest(path),"config_sha256":digest(CONFIG),"cuda_initialized":False,"training_steps":0,"environment_resets":0,"environment_steps":0,"model_predictions":0,"boundary":c["boundary"]}
    if RESULT.exists(): raise FileExistsError(f"refusing overwrite: {RESULT}")
    RESULT.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["classification"]=="PASS" else 1)
if __name__=="__main__": main()
