#!/usr/bin/env python3
"""CPU-only audit of native wrist demonstrations for Phase 4C."""
from pathlib import Path
import hashlib,json,h5py,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]; CONFIG=ROOT/"configs/dataset/phase4c_libero_goal_wrist_adaptation_v1.yaml"; RESULT=ROOT/"results/phase4c_libero_goal_wrist_data_contract_v1.yaml"
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 c=yaml.safe_load(CONFIG.read_text()); path=ROOT/c['dataset']['path']; assert digest(path)==c['dataset']['sha256']
 train=c['split']['train_episodes']; val=c['split']['validation_episodes']; assert len(train)==40 and len(val)==10 and not set(train)&set(val) and sorted(train+val)==list(range(50))
 stats={}; cross=[]
 with h5py.File(path,'r') as f:
  for split,ids in [('train',train),('validation',val)]:
   frames=0; sums=np.zeros(3); squares=np.zeros(3); mins=np.full(3,255); maxs=np.zeros(3); actions=[]
   for e in ids:
    g=f[f'data/demo_{e}']; wrist=g[c['dataset']['image_source']]; agent=g[c['dataset']['paired_reference_source']]; action=g[c['dataset']['action_source']]
    n=len(action); assert wrist.shape==agent.shape==(n,128,128,3) and wrist.dtype==agent.dtype==np.uint8 and action.shape==(n,7)
    w=wrist[()]; a=agent[()]; flat=w.reshape(-1,3).astype(np.float64); frames+=n; sums+=flat.sum(0); squares+=(flat*flat).sum(0); mins=np.minimum(mins,flat.min(0)); maxs=np.maximum(maxs,flat.max(0)); actions.append(action[()]); cross.append(float(np.mean(np.abs(w.astype(np.int16)-a.astype(np.int16)))))
   pixels=frames*128*128; mean=sums/pixels; std=np.sqrt(squares/pixels-mean*mean); acts=np.concatenate(actions)
   stats[split]={'episodes':len(ids),'frames':frames,'wrist_rgb_mean':mean.tolist(),'wrist_rgb_std':std.tolist(),'wrist_rgb_min':mins.astype(int).tolist(),'wrist_rgb_max':maxs.astype(int).tolist(),'actions_finite':bool(np.isfinite(acts).all())}
 checks={'dataset_hash':True,'split_exact':True,'train_frames':stats['train']['frames']==3571,'validation_frames':stats['validation']['frames']==889,'wrist_nonconstant':all(x>10 for x in stats['train']['wrist_rgb_std']),'views_distinct':float(np.mean(cross))>10,'actions_finite':stats['train']['actions_finite'] and stats['validation']['actions_finite'],'canonical_revision':c['training_candidate']['base_revision']=='fa5ae1e7509348889295bba8e08621d8b55e9baf','final_tasks_untouched':c['final_tasks_untouched']==[1,2,4,5,9]}
 result={'classification':'PASS' if all(checks.values()) else 'FAIL','experiment_id':c['experiment_id'],'checks':checks,'splits':stats,'mean_episode_cross_view_mae':float(np.mean(cross)),'dataset_sha256':digest(path),'config_sha256':digest(CONFIG),'cuda_calls':0,'environment_resets':0,'environment_steps':0,'model_predictions':0,'boundary':c['boundary']}
 if RESULT.exists(): raise FileExistsError(f'refusing overwrite: {RESULT}')
 RESULT.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2))
 if result['classification']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
