#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import h5py,numpy as np,yaml
from PIL import Image
root=Path(__file__).resolve().parents[1]; cfg=yaml.safe_load((root/'configs/eval/openvla_phase3b_qlora_step20_fullval889_v1.yaml').read_text())
source=root/cfg['dataset']['path']; output=root/cfg['fixture_dir']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
if output.exists(): raise FileExistsError(f'refusing overwrite: {output}')
items=[]; output.mkdir(parents=True)
with h5py.File(source,'r') as f:
 for episode in cfg['validation_episodes']:
  g=f[f'data/demo_{episode}']
  for timestep in range(len(g['actions'])):
   image=np.asarray(g['obs/agentview_rgb'][timestep]); action=np.asarray(g['actions'][timestep],dtype=np.float64); action[-1]=1-np.clip(action[-1],0,1)
   path=output/f'demo{episode:02d}_t{timestep:03d}.png'; Image.fromarray(image,'RGB').save(path)
   items.append({'episode':episode,'timestep':timestep,'image_path':str(path.relative_to(root)),'image_sha256':sha(path),'openvla_training_action':action.tolist()})
if len(items)!=cfg['expected_validation_frames']: raise ValueError(f'expected 889, got {len(items)}')
manifest={'classification':'PASS','source_sha256':sha(source),'episodes':cfg['validation_episodes'],'frames':len(items),'items':items,'boundary':'Complete frozen offline validation fixture; no environment states or rewards.'}
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps({'classification':'PASS','frames':len(items),'manifest':str((output/'manifest.json').relative_to(root)),'manifest_sha256':sha(output/'manifest.json')}))
