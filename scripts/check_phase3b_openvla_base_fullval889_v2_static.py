#!/usr/bin/env python3
from pathlib import Path
import ast,json,yaml
root=Path(__file__).resolve().parents[1]; script=root/'scripts/evaluate_phase3b_openvla_base_fullval889_v2.py'; config=root/'configs/eval/openvla_phase3b_base_fullval889_v2.yaml'
text=script.read_text(); ast.parse(text); cfg=yaml.safe_load(config.read_text()); manifest=json.loads((root/cfg['fixture_dir']/'manifest.json').read_text())
assert len(manifest['items'])==889 and cfg['expected_validation_frames']==889
assert 'wrapped=model; patches=' in text and 'model.base_model.model' not in text and 'PeftModel' not in text
assert cfg['output_dir'].endswith('base_fullval889_v2_seed20260827') and not (root/cfg['output_dir']).exists()
print({'classification':'PASS','experiment_id':cfg['experiment_id']+'_static','direct_base_model_alignment':True,'cuda_initialized':False,'model_loaded':False})
