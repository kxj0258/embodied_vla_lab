#!/usr/bin/env python3
"""Phase 4D base entrypoint using the frozen Phase 3B full-validation implementation."""
from pathlib import Path
import evaluate_phase3b_openvla_base_fullval889_v2 as evaluator
evaluator.CONFIG=Path(__file__).resolve().parents[1]/'configs/eval/openvla_phase4d_wrist_base_fullval889_v1.yaml'
if __name__=='__main__': evaluator.main()
