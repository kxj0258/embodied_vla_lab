#!/usr/bin/env python3
"""Phase 4C entrypoint reusing the frozen Phase 3B trainer with a wrist-image fixture/config."""
from pathlib import Path
import train_phase3b_openvla_qlora_smoke_v3 as trainer
trainer.CONFIG=Path(__file__).resolve().parents[1]/'configs/train/openvla_phase4c_wrist_qlora_train_smoke_v1.yaml'
if __name__=='__main__': trainer.main()
