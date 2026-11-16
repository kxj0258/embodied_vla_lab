#!/usr/bin/env python3
from pathlib import Path
import ast,hashlib,json,yaml
root=Path(__file__).resolve().parents[1]; config=root/'configs/eval/openvla_phase3b_base_fullval889_v1.yaml'; script=root/'scripts/evaluate_phase3b_openvla_base_fullval889.py'
cfg=yaml.safe_load(config.read_text()); text=script.read_text(); ast.parse(text)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
manifest_path=root/cfg['fixture_dir']/'manifest.json'; manifest=json.loads(manifest_path.read_text())
assert sha(manifest_path)==cfg['fixture_manifest_sha256'] and len(manifest['items'])==889
assert 'PeftModel' not in text and "'adapter_sha256':None" in text
assert 'torch.no_grad()' in text and 'training_executed' in text and 'environment_executed' in text
assert not (root/cfg['output_dir']).exists()
print({'classification':'PASS','experiment_id':cfg['experiment_id']+'_static','frames':889,'base_only':True,'cuda_initialized':False,'model_loaded':False})
