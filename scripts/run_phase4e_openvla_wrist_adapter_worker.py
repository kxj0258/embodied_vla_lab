#!/usr/bin/env python3
"""Config-frozen Phase 4C wrist-adapter persistent inference worker."""
from pathlib import Path

import run_phase3b_openvla_qlora_persistent_worker as worker

ROOT = Path(__file__).resolve().parents[1]
worker.ADAPTER = ROOT / "outputs/train/phase4c_openvla_wrist_qlora_rank8_train20_val10_v1_seed20260829/adapter"
worker.ADAPTER_SHA256 = "fad1e3201dacc5e1136f45b5df3b06943d83354a34c1079d620c4b1abcf1ae7a"

if __name__ == "__main__":
    worker.main()
