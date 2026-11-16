#!/usr/bin/env python3
from pathlib import Path
import ast, hashlib, json, yaml
root=Path(__file__).resolve().parents[1]
config_path=root/'configs/train/openvla_phase3b_qlora_train_smoke_v1.yaml'
script_path=root/'scripts/train_phase3b_openvla_qlora_smoke.py'
cfg=yaml.safe_load(config_path.read_text()); text=script_path.read_text(); ast.parse(text)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()
manifest_path=root/cfg['smoke_fixture']; assert sha(manifest_path)==cfg['smoke_fixture_sha256']
manifest=json.loads(manifest_path.read_text()); assert manifest['source_sha256']==cfg['dataset']['sha256']
train=set(cfg['split']['train_episodes']); val=set(cfg['split']['validation_episodes'])
assert len(train)==40 and len(val)==10 and not train.intersection(val) and train|val==set(range(50))
items=manifest['items']; train_items=[x for x in items if x['split']=='train']; val_items=[x for x in items if x['split']=='validation_smoke']
assert manifest['train_pool_frames']==3571 and len(train_items)==20 and len(val_items)==10
assert len({(x['episode'],x['timestep']) for x in train_items})==20
assert {x['episode'] for x in val_items}==val
for item in items: assert sha(root/item['image_path'])==item['image_sha256']
for token in ('optimizer.step()','save_pretrained','optimizer_and_rng.pt','initial_validation','final_validation','checkpoint_files'):
 assert token in text, token
assert not (root/cfg['output_dir']).exists(), 'refusing existing formal output'
print({'classification':'PASS','experiment_id':cfg['experiment_id']+'_static','train_episodes':40,'train_pool_frames':3571,'validation_episodes':10,'full_validation_frames':889,'validation_smoke_frames':10,'training_steps':20,'cuda_initialized':False,'model_loaded':False})
