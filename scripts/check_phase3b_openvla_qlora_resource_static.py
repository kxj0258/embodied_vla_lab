#!/usr/bin/env python3
"""No-CUDA static validation for the first QLoRA forward/backward gate."""
import ast,hashlib,json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]; CONFIG=ROOT/'configs/train/openvla_phase3b_qlora_resource_gate.yaml'; ENTRY=ROOT/'scripts/check_phase3b_openvla_qlora_resource_gate.py'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cfg=yaml.safe_load(CONFIG.read_text()); source=ENTRY.read_text(); ast.parse(source)
 assert cfg['lora']=={'rank':8,'alpha':8,'dropout':0.0,'target_modules':'all-linear','initialization':'gaussian'}
 assert cfg['batch_size']==1 and cfg['forward_steps']==1 and cfg['backward_steps']==1 and cfg['optimizer_steps']==0
 assert not (ROOT/cfg['output_dir']).exists(),'formal resource output exists'
 fixture_path=ROOT/cfg['fixture']; fixture=json.loads(fixture_path.read_text())
 assert fixture['source_sha256']=='387fc10747696b80dea6ed8d7f2beaa162bf92ae11750b241073cbd33aac73d5'
 assert fixture['image_sha256']=='d4c9852bcb00f084c104ba1381579720a154796eecbde26669d53a97a0278dbd'
 assert fixture['gripper_transform']=='1-clip(raw,0,1)' and fixture['openvla_training_action'][-1]==1.0
 for contract in ['prepare_model_for_kbit_training','target_modules=lc[\'target_modules\']','labels[:,:-(7+1)]=-100','labels[:,-1]=-100','loss.backward()','optimizer_steps\':0','checkpoint_created\':False']:
  assert contract in source,contract
 print('classification: PASS'); print('cuda_initialized: false'); print('model_loaded: false'); print('forward_steps: 0'); print('backward_steps: 0'); print('qlora resource static contract: PASS'); print('fixture_manifest_sha256:',sha(fixture_path))
if __name__=='__main__': main()
