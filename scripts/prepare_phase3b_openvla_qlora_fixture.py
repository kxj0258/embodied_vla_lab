#!/usr/bin/env python3
"""Export one real LIBERO image/action pair for the QLoRA resource gate."""
import hashlib, json
from pathlib import Path
import h5py, numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'libero/libero/datasets/libero_goal/turn_on_the_stove_demo.hdf5'
OUTPUT=ROOT/'outputs/reference/phase3b_openvla_qlora_fixture_demo0_t000'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 if OUTPUT.exists(): raise FileExistsError(f'refusing overwrite: {OUTPUT}')
 with h5py.File(SOURCE,'r') as f:
  g=f['data/demo_0']; image=g['obs/agentview_rgb'][0]; raw=np.asarray(g['actions'][0],dtype=np.float64)
 if image.shape!=(128,128,3) or image.dtype!=np.uint8 or raw.shape!=(7,) or not np.isfinite(raw).all(): raise ValueError('fixture contract mismatch')
 transformed=raw.copy(); transformed[-1]=1.0-np.clip(transformed[-1],0.0,1.0)
 OUTPUT.mkdir(parents=True); image_path=OUTPUT/'agentview_rgb.png'; Image.fromarray(image,mode='RGB').save(image_path)
 result={'classification':'PASS','source':str(SOURCE.relative_to(ROOT)),'source_sha256':sha(SOURCE),'demo':'demo_0','timestep':0,
  'image_path':str(image_path.relative_to(ROOT)),'image_sha256':sha(image_path),'image_shape':[128,128,3],'image_dtype':'uint8',
  'image_orientation':'stored_training_orientation_no_rotation','raw_environment_action':raw.tolist(),'openvla_training_action':transformed.tolist(),
  'gripper_transform':'1-clip(raw,0,1)','instruction':'turn on the stove'}
 manifest=OUTPUT/'fixture_manifest.json'; manifest.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))

if __name__=='__main__': main()
