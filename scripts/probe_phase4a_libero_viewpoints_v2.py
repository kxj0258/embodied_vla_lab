#!/usr/bin/env python3
"""Corrected v2 entrypoint; preserve the immutable v1 failed implementation and artifacts."""
from pathlib import Path

import probe_phase4a_libero_viewpoints as implementation

implementation.CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/eval/phase4a_openvla_viewpoint_camera_gate_v2.yaml"
)

if __name__ == "__main__":
    implementation.main()
