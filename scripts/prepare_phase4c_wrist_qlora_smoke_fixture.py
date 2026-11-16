#!/usr/bin/env python3
"""Export a deterministic 20-train/10-validation native wrist-image fixture."""
from pathlib import Path
import hashlib,json,h5py,numpy as np,yaml
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; CONFIG=ROOT/'configs/dataset/phase4c_libero_goal_wrist_adaptation_v1.yaml'; OUTPUT=ROOT/'outputs/reference/phase4c_wrist_qlora_train20_val10_fixture_v1_seed20260828'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 if OUTPUT.exists(): raise FileExistsError(f'refusing overwrite: {OUTPUT}')
 c=yaml.safe_load(CONFIG.read_text()); source=ROOT/c['dataset']['path']; assert digest(source)==c['dataset']['sha256']; items=[]; sample_seed=20260828
 with h5py.File(source,'r') as f:
  refs=[(e,t) for e in c['split']['train_episodes'] for t in range(len(f[f'data/demo_{e}/actions']))]; rng=np.random.default_rng(sample_seed); selected=[refs[i] for i in rng.choice(len(refs),size=20,replace=False)]; chosen=[('train',e,t) for e,t in selected]+[('validation_smoke',e,0) for e in c['split']['validation_episodes']]
  OUTPUT.mkdir(parents=True)
  for i,(split,e,t) in enumerate(chosen):
   g=f[f'data/demo_{e}']; image=np.asarray(g[c['dataset']['image_source']][t]); action=np.asarray(g['actions'][t],dtype=np.float64); assert image.shape==(128,128,3) and image.dtype==np.uint8
   action[-1]=1.0-np.clip(action[-1],0.,1.); path=OUTPUT/f'{i:03d}_{split}_demo{e}_t{t}_eye_in_hand.png'; Image.fromarray(image,'RGB').save(path); items.append({'split':split,'episode':e,'timestep':t,'view':'eye_in_hand','image_path':str(path.relative_to(ROOT)),'image_sha256':digest(path),'openvla_training_action':action.tolist()})
 manifest={'classification':'PASS','source':str(source.relative_to(ROOT)),'source_sha256':digest(source),'image_source':c['dataset']['image_source'],'sample_seed':sample_seed,'train_pool_frames':len(refs),'train_samples':20,'validation_smoke_samples':10,'items':items,'boundary':'Fixed development wrist-image smoke subset; offline metrics are not environment success.'}; mp=OUTPUT/'manifest.json'; mp.write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps({'classification':'PASS','manifest':str(mp.relative_to(ROOT)),'manifest_sha256':digest(mp),'train_pool_frames':len(refs),'items':30,'environment_steps':0,'model_predictions':0},indent=2))
if __name__=='__main__': main()
