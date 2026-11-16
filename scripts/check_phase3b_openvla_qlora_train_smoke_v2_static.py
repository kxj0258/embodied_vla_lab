#!/usr/bin/env python3
from pathlib import Path
import ast, yaml
root=Path(__file__).resolve().parents[1]
wrapper=root/'scripts/train_phase3b_openvla_qlora_smoke_v2.py'; implementation=root/'scripts/train_phase3b_openvla_qlora_smoke.py'
cfg=yaml.safe_load((root/'configs/train/openvla_phase3b_qlora_train_smoke_v2.yaml').read_text())
tree=ast.parse(wrapper.read_text()); ast.parse(implementation.read_text())
calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
assert any(isinstance(n.func,ast.Attribute) and n.func.attr=='main' for n in calls)
assert cfg['training_steps']==20 and cfg['output_dir'].endswith('train20_val10_v2_seed20260827')
assert not (root/cfg['output_dir']).exists()
print({'classification':'PASS','experiment_id':cfg['experiment_id']+'_static','main_invocation':True,'cuda_initialized':False,'model_loaded':False})
