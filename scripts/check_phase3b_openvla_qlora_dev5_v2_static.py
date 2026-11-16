#!/usr/bin/env python3
from pathlib import Path
import ast,yaml
root=Path(__file__).resolve().parents[1]; cfg=yaml.safe_load((root/'configs/eval/openvla_libero_phase3b_qlora_step20_dev5_v2.yaml').read_text())
worker=(root/cfg['model']['worker']).read_text(); ast.parse(worker)
assert cfg['protocol']['init_state_indices']==[0,1,2,3,4] and cfg['protocol']['episodes']==5
assert cfg['output_dir'].endswith('dev5_v2_init0to4_seed20260827') and not (root/cfg['output_dir']).exists()
assert 'PeftModel.from_pretrained' in worker
print({'classification':'PASS','experiment_id':cfg['experiment_id']+'_static','compute_contract_changed':False,'required_host_free_vram_mib':8000,'cuda_initialized':False,'environment_created':False})
