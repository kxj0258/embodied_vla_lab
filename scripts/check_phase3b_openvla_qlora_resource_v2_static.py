#!/usr/bin/env python3
"""Static regression for v2 serialization fix after successful compute reached v1 writer."""
import ast
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; CONFIG=ROOT/'configs/train/openvla_phase3b_qlora_resource_gate_v2.yaml'; ENTRY=ROOT/'scripts/check_phase3b_openvla_qlora_resource_gate.py'
def main():
 cfg=yaml.safe_load(CONFIG.read_text()); source=ENTRY.read_text(); ast.parse(source)
 assert cfg['experiment_id']=='phase3b_openvla_qlora_rank8_batch1_onebackward_v2_seed20260827'
 assert not (ROOT/cfg['output_dir']).exists(),'v2 formal output exists'
 assert 'output_dir=ROOT/cfg[\'output_dir\']' in source and 'model_output=model(**inputs)' in source
 assert "loss=model_output.loss" in source and "logits=model_output.logits" in source
 assert "output_dir.mkdir(parents=True)" in source
 assert "with torch.autocast('cuda',dtype=torch.bfloat16)" in source and 'loss.backward()' in source
 print('classification: PASS'); print('cuda_initialized: false'); print('model_loaded: false'); print('v2 variable-shadow serialization regression: PASS')
if __name__=='__main__': main()
