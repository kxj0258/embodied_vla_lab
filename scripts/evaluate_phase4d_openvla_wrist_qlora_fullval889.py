#!/usr/bin/env python3
"""Phase 4D adapter entrypoint using the frozen Phase 3B full-validation implementation."""
from pathlib import Path
import evaluate_phase3b_openvla_qlora_fullval889 as evaluator
evaluator.CONFIG=Path(__file__).resolve().parents[1]/'configs/eval/openvla_phase4d_wrist_qlora_step20_fullval889_v1.yaml'
if __name__=='__main__': evaluator.main()
