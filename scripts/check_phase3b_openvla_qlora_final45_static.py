#!/usr/bin/env python3
from pathlib import Path
import ast,yaml
root=Path(__file__).resolve().parents[1]; cfg=yaml.safe_load((root/'configs/eval/openvla_libero_phase3b_qlora_step20_final_heldout45_v1.yaml').read_text())
worker=(root/cfg['model']['worker']).read_text(); ast.parse(worker); ast.parse((root/'scripts/evaluate_phase3a_openvla_libero_final45.py').read_text())
assert cfg['protocol']['init_state_indices']==list(range(5,50)) and cfg['protocol']['development_excluded_init_state_indices']==list(range(5))
assert cfg['protocol']['episodes']==45 and cfg['model']['adapter_sha256']=='6b4fce2e469048c17650f7cc029f3154dc0742bf56d49651d6cfd369bdd2735c'
assert 'PeftModel.from_pretrained' in worker and not (root/cfg['output_dir']).exists()
print({'classification':'PASS','experiment_id':cfg['experiment_id']+'_static','heldout_states':45,'development_excluded':[0,1,2,3,4],'required_host_free_vram_mib':8000,'cuda_initialized':False,'environment_created':False})
