#!/usr/bin/env python3
from pathlib import Path
import ast, yaml
root=Path(__file__).resolve().parents[1]
script=root/'scripts/check_phase3b_openvla_qlora_optimizer_gate_v2.py'
cfg=yaml.safe_load((root/'configs/train/openvla_phase3b_qlora_optimizer_gate_v2.yaml').read_text())
text=script.read_text(); ast.parse(text)
assert cfg['optimizer_steps']==1 and cfg['max_grad_norm']==1.0
assert cfg['output_dir'].endswith('oneupdate_v2_seed20260827')
for token in ('update_grad_max','named_parameters()','clip_grad_norm_','optimizer.step()','optimizer_state_bytes','parameter_update_max_abs'):
 assert token in text, token
assert 'first_param=next' not in text
assert not (root/cfg['output_dir']).exists(), 'refusing existing formal output'
print({'classification':'PASS','experiment_id':cfg['experiment_id']+'_static','cuda_initialized':False,'model_loaded':False,'optimizer_steps':0})
