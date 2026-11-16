#!/usr/bin/env python3
from pathlib import Path
import ast,hashlib,json,yaml
root=Path(__file__).resolve().parents[1]; config=root/'configs/eval/openvla_phase3b_qlora_step20_fullval889_v1.yaml'; script=root/'scripts/evaluate_phase3b_openvla_qlora_fullval889.py'
cfg=yaml.safe_load(config.read_text()); text=script.read_text(); ast.parse(text)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
manifest_path=root/cfg['fixture_dir']/'manifest.json'; manifest=json.loads(manifest_path.read_text())
assert sha(manifest_path)==cfg['fixture_manifest_sha256'] and manifest['frames']==889
assert len(manifest['items'])==889 and {x['episode'] for x in manifest['items']}==set(cfg['validation_episodes'])
assert all(sha(root/x['image_path'])==x['image_sha256'] for x in manifest['items'])
assert sha(root/cfg['adapter_dir']/'adapter_model.safetensors')==cfg['adapter_sha256']
for token in ('PeftModel.from_pretrained','torch.no_grad()','training_executed','environment_executed','checkpoint_selected'):
 assert token in text
assert not (root/cfg['output_dir']).exists()
print({'classification':'PASS','experiment_id':cfg['experiment_id']+'_static','frames':889,'episodes':10,'adapter_hash':cfg['adapter_sha256'],'cuda_initialized':False,'model_loaded':False})
