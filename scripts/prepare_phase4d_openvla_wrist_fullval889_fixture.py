#!/usr/bin/env python3
"""Export all 889 frozen validation wrist frames for paired Phase 4D validation."""
from pathlib import Path
import hashlib,json,h5py,numpy as np,yaml
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'configs/dataset/phase4c_libero_goal_wrist_adaptation_v1.yaml'; OUT=ROOT/'outputs/reference/phase4d_openvla_wrist_fullval889_fixture_v1'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 if OUT.exists(): raise FileExistsError(f'refusing overwrite: {OUT}')
 c=yaml.safe_load(DATA.read_text()); source=ROOT/c['dataset']['path']; assert digest(source)==c['dataset']['sha256']; items=[]; OUT.mkdir(parents=True)
 with h5py.File(source,'r') as f:
  for e in c['split']['validation_episodes']:
   g=f[f'data/demo_{e}']
   for t in range(len(g['actions'])):
    image=np.asarray(g[c['dataset']['image_source']][t]); action=np.asarray(g['actions'][t],dtype=np.float64); assert image.shape==(128,128,3) and image.dtype==np.uint8 and action.shape==(7,) and np.isfinite(action).all(); action[-1]=1-np.clip(action[-1],0,1)
    path=OUT/f'demo{e:02d}_t{t:03d}_eye_in_hand.png'; Image.fromarray(image,'RGB').save(path); items.append({'episode':e,'timestep':t,'view':'eye_in_hand','image_path':str(path.relative_to(ROOT)),'image_sha256':digest(path),'openvla_training_action':action.tolist()})
 assert len(items)==889
 manifest={'classification':'PASS','source_sha256':digest(source),'image_source':c['dataset']['image_source'],'episodes':c['split']['validation_episodes'],'frames':889,'items':items,'boundary':'Complete frozen wrist offline validation fixture; no environment states, success or reward.'}; mp=OUT/'manifest.json'; mp.write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps({'classification':'PASS','frames':889,'manifest':str(mp.relative_to(ROOT)),'manifest_sha256':digest(mp),'environment_steps':0,'model_predictions':0},indent=2))
if __name__=='__main__': main()
