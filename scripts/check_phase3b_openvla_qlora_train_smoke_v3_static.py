#!/usr/bin/env python3
from pathlib import Path
import ast, yaml
root=Path(__file__).resolve().parents[1]
script=root/'scripts/train_phase3b_openvla_qlora_smoke_v3.py'; config=root/'configs/train/openvla_phase3b_qlora_train_smoke_v3.yaml'
text=script.read_text(); tree=ast.parse(text); cfg=yaml.safe_load(config.read_text())
assert cfg['training_steps']==20 and cfg['output_dir'].endswith('train20_val10_v3_seed20260827')
for token in ('del out,pred,gt,mask,inputs,labels','gc.collect()','torch.cuda.empty_cache()','optimizer.step()','save_pretrained'):
 assert token in text, token
assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='main' for n in ast.walk(tree))
assert not (root/cfg['output_dir']).exists()
print({'classification':'PASS','experiment_id':cfg['experiment_id']+'_static','phase_cleanup_verified':True,'main_invocation':True,'cuda_initialized':False,'model_loaded':False})
