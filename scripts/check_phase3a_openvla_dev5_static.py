#!/usr/bin/env python3
"""Static validation for the bounded OpenVLA development rollout."""
from __future__ import annotations

import ast
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/eval/openvla_libero_phase3a_dev5.yaml"
ENTRYPOINT = ROOT / "scripts/evaluate_phase3a_openvla_libero_dev5.py"


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    assert config["protocol"]["init_state_indices"] == [0, 1, 2, 3, 4]
    assert config["protocol"]["episodes"] == 5
    assert config["protocol"]["episode_horizon"] == 600
    assert config["model"]["worker"] == "scripts/run_phase3a_openvla_persistent_worker.py"
    assert not (ROOT / config["output_dir"]).exists(), "formal dev5 output already exists"
    source = ENTRYPOINT.read_text()
    ast.parse(source, filename=str(ENTRYPOINT))
    for contract in [
        'indices != [0, 1, 2, 3, 4]',
        'worker.stdin.write(json.dumps({"command": "shutdown"})',
        'if terminated_value or truncated_value',
        'worker.terminate()',
        'env.close()',
        '"success_count": successes',
    ]:
        assert contract in source, contract
    print("classification: PASS")
    print("environment_constructed: false")
    print("cuda_initialized: false")
    print("model_loaded: false")
    print("episodes_executed: 0")
    print("bounded dev5 static contract: PASS")


if __name__ == "__main__":
    main()
