#!/usr/bin/env python3
from pathlib import Path
import ast,hashlib,yaml
root=Path(__file__).resolve().parents[1]; config=root/'configs/eval/openvla_libero_phase3b_qlora_step20_single_episode_v1.yaml'; worker=root/'scripts/run_phase3b_openvla_qlora_persistent_worker.py'
cfg=yaml.safe_load(config.read_text()); text=worker.read_text(); ast.parse(text)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
assert cfg['protocol']['episodes']==1 and cfg['protocol']['init_state_index']==0 and cfg['protocol']['episode_horizon']==600
assert sha(root/cfg['model']['adapter_dir']/'adapter_model.safetensors')==cfg['model']['adapter_sha256']
for token in ('PeftModel.from_pretrained','ADAPTER_SHA256','is_trainable=False','predict_action'):
 assert token in text
assert not (root/cfg['output_dir']).exists()
print({'classification':'PASS','experiment_id':cfg['experiment_id']+'_static','episodes':1,'init_state_index':0,'adapter_hash':cfg['model']['adapter_sha256'],'cuda_initialized':False,'environment_created':False})
