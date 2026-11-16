#!/usr/bin/env python3
"""V2 entrypoint: execute the frozen smoke implementation with a new immutable config/output ID."""
import os
from pathlib import Path
import train_phase3b_openvla_qlora_smoke as implementation
implementation.CONFIG=Path(__file__).resolve().parents[1]/'configs/train/openvla_phase3b_qlora_train_smoke_v2.yaml'
if __name__=='__main__':
 os.environ.setdefault('HF_HUB_OFFLINE','1')
 os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
 os.environ.setdefault('WANDB_MODE','disabled')
 implementation.main()
