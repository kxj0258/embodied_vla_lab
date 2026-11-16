#!/usr/bin/env python3
"""Static validation of the preregistered OpenVLA base-policy held-out final."""
from pathlib import Path
import ast, yaml

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/eval/openvla_libero_phase3a_final_heldout45.yaml'
ENTRY=ROOT/'scripts/evaluate_phase3a_openvla_libero_final45.py'


def main():
    cfg=yaml.safe_load(CONFIG.read_text()); protocol=cfg['protocol']; source=ENTRY.read_text(); ast.parse(source)
    assert protocol['kind']=='preregistered_base_policy_final'
    assert protocol['init_state_indices']==list(range(5,50))
    assert protocol['development_excluded_init_state_indices']==list(range(5))
    assert protocol['episodes']==45 and protocol['episode_horizon']==600
    assert cfg['model']['revision']=='fa5ae1e7509348889295bba8e08621d8b55e9baf'
    assert not (ROOT/cfg['output_dir']).exists(), 'formal final output already exists'
    for contract in ["indices != list(range(5,50))","episode_index=5","'success_total':45","worker.terminate()","env.close()"]:
        assert contract in source,contract
    print('classification: PASS')
    print('environment_constructed: false')
    print('cuda_initialized: false')
    print('model_loaded: false')
    print('heldout_episodes_executed: 0')
    print('preregistered final45 static contract: PASS')


if __name__=='__main__': main()
