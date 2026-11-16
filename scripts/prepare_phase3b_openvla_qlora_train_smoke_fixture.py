#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
import h5py, numpy as np, yaml
from PIL import Image
root=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((root/'configs/train/openvla_phase3b_qlora_train_smoke_v1.yaml').read_text())
source=root/cfg['dataset']['path']; output=root/'outputs/reference/phase3b_openvla_qlora_train20_val10_fixture_v1_seed20260828'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()
if output.exists(): raise FileExistsError(f'refusing overwrite: {output}')
items=[]
with h5py.File(source,'r') as f:
 train_refs=[(e,t) for e in cfg['split']['train_episodes'] for t in range(len(f[f'data/demo_{e}/actions']))]
 rng=np.random.default_rng(cfg['sample_seed']); selected=[train_refs[i] for i in rng.choice(len(train_refs),size=cfg['training_steps'],replace=False)]
 refs=[('train',e,t) for e,t in selected]+[('validation_smoke',e,0) for e in cfg['split']['validation_episodes']]
 output.mkdir(parents=True)
 for index,(split,e,t) in enumerate(refs):
  g=f[f'data/demo_{e}']; image=np.asarray(g['obs/agentview_rgb'][t]); action=np.asarray(g['actions'][t],dtype=np.float64)
  action[-1]=1.0-np.clip(action[-1],0.0,1.0)
  path=output/f'{index:03d}_{split}_demo{e}_t{t}.png'; Image.fromarray(image,'RGB').save(path)
  items.append({'split':split,'episode':e,'timestep':t,'image_path':str(path.relative_to(root)),'image_sha256':sha(path),'openvla_training_action':action.tolist()})
manifest={'classification':'PASS','source':str(source.relative_to(root)),'source_sha256':sha(source),'sample_seed':cfg['sample_seed'],'train_pool_frames':len(train_refs),'train_samples':20,'validation_smoke_samples':10,'items':items,'boundary':'Fixed smoke subset only; full validation remains 889 frames.'}
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({**{k:v for k,v in manifest.items() if k!='items'},'manifest':str((output/'manifest.json').relative_to(root)),'manifest_sha256':sha(output/'manifest.json')}))
