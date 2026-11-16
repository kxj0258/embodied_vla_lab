#!/usr/bin/env python3
from pathlib import Path
import ast,yaml
root=Path(__file__).resolve().parents[1]; cfg=yaml.safe_load((root/'configs/eval/openvla_libero_phase3b_qlora_step20_dev5_v1.yaml').read_text())
ast.parse((root/'scripts/evaluate_phase3a_openvla_libero_dev5.py').read_text()); worker=(root/cfg['model']['worker']).read_text(); ast.parse(worker)
assert cfg['protocol']['init_state_indices']==[0,1,2,3,4] and cfg['protocol']['episodes']==5
assert cfg['model']['adapter_sha256']=='6b4fce2e469048c17650f7cc029f3154dc0742bf56d49651d6cfd369bdd2735c'
assert 'PeftModel.from_pretrained' in worker and not (root/cfg['output_dir']).exists()
print({'classification':'PASS','experiment_id':cfg['experiment_id']+'_static','init_states':[0,1,2,3,4],'episodes':5,'cuda_initialized':False,'environment_created':False})
