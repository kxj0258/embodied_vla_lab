#!/usr/bin/env python
"""Read-only Phase 2A LIBERO single-task static contract checker.

This checker deliberately does not import LIBERO, construct an environment, initialize
CUDA, render, train, evaluate, download, or write files. It reads the frozen HDF5 schema.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import torch
import h5py
import numpy as np
import yaml

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/env/libero_goal_turn_on_stove_phase2a.yaml"
LIBERO = ROOT / "libero"
BDDL_ROOT = LIBERO / "libero/libero/bddl_files"
INIT_ROOT = LIBERO / "libero/libero/init_files"

EXPECTED_PIN = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
EXPECTED_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_90": 90,
    "libero_10": 10,
}
EXPECTED_BDDL_SHA256 = "cbd2a0514d53226c5d09e17af40b22935d4cb6ff8fdaff0d55f555de1adecc13"
EXPECTED_INIT_SHA256 = "ed121db31d749647d70741c4c4ef0bd8e0876663986578795968e962d9b91340"
EXPECTED_DEMO_SHA256 = "387fc10747696b80dea6ed8d7f2beaa162bf92ae11750b241073cbd33aac73d5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    print("classification: PASS (pending all assertions)")
    print("mode: static + HDF5 read-only; LIBERO import/environment/CUDA/network/write disabled")

    pin = subprocess.run(
        ["git", "-C", str(LIBERO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert pin == EXPECTED_PIN
    print(f"libero pin: {pin}")
    print(f"robosuite: {version('robosuite')}; mujoco: {version('mujoco')}")

    for suite, expected in EXPECTED_COUNTS.items():
        bddl_count = len(list((BDDL_ROOT / suite).glob("*.bddl")))
        init_count = len(list((INIT_ROOT / suite).glob("*.pruned_init")))
        assert (bddl_count, init_count) == (expected, expected)
        print(f"{suite}: bddl={bddl_count}, pruned_init={init_count}")

    bddl = BDDL_ROOT / cfg["problem_folder"] / cfg["bddl_file"]
    init_states_path = INIT_ROOT / cfg["problem_folder"] / cfg["init_states_file"]
    bddl_text = bddl.read_text(encoding="utf-8")
    assert "(:language Turn on the stove)" in bddl_text
    assert "(And (Turnon flat_stove_1))" in bddl_text
    assert sha256(bddl) == EXPECTED_BDDL_SHA256

    init_states = torch.load(init_states_path, map_location="cpu", weights_only=False)
    assert tuple(init_states.shape) == (50, 79)
    assert type(init_states).__module__ == "numpy"
    assert str(init_states.dtype) == "float64"
    assert sha256(init_states_path) == EXPECTED_INIT_SHA256
    print(f"task: {cfg['suite']}/{cfg['task_name']} (index {cfg['task_index']})")
    print(f"bddl sha256: {EXPECTED_BDDL_SHA256}")
    print(
        f"init states: {type(init_states).__name__} {tuple(init_states.shape)} "
        f"{init_states.dtype}; sha256: {EXPECTED_INIT_SHA256}"
    )

    assert cfg["action"]["shape"] == [7]
    assert cfg["control_frequency_hz"] == 20
    assert cfg["evaluation_horizon_policy_steps"] == 600
    assert cfg["physics_settle_steps"] == 5

    dataset_root = LIBERO / "libero/datasets"
    demo = dataset_root / cfg["demonstration"]["expected_relative_path"]
    assert demo.stat().st_size == 447509922
    assert sha256(demo) == EXPECTED_DEMO_SHA256
    with h5py.File(demo, "r") as file:
        data = file["data"]
        demos = sorted(data, key=lambda name: int(name.split("_")[-1]))
        assert len(demos) == 50
        assert int(data.attrs["num_demos"]) == 50
        assert int(data.attrs["total"]) == 4460
        lengths = []
        action_min = None
        action_max = None
        for name in demos:
            group = data[name]
            actions = group["actions"][()]
            lengths.append(len(actions))
            assert actions.shape[1:] == (7,)
            assert actions.dtype == np.float64
            assert np.isfinite(actions).all()
            assert int(group["rewards"][()].sum()) == 1
            assert int(group["dones"][()].sum()) == 1
            action_min = actions.min(0) if action_min is None else np.minimum(action_min, actions.min(0))
            action_max = actions.max(0) if action_max is None else np.maximum(action_max, actions.max(0))
        assert sum(lengths) == 4460
        assert (min(lengths), max(lengths)) == (75, 119)
        assert float(action_min.min()) >= -1.0
        assert float(action_max.max()) <= 1.0
        first = data[demos[0]]
        assert first["obs/agentview_rgb"].shape[1:] == (128, 128, 3)
        assert first["obs/eye_in_hand_rgb"].shape[1:] == (128, 128, 3)
        assert first["obs/gripper_states"].shape[1:] == (2,)
        assert first["obs/joint_states"].shape[1:] == (7,)
    print(f"demonstration: PASS 50 episodes / 4460 frames / sha256 {EXPECTED_DEMO_SHA256}")

    default_config = Path.home() / ".libero/config.yaml"
    print(f"LIBERO_CONFIG_PATH: {os.environ.get('LIBERO_CONFIG_PATH', '<unset>')}")
    print(f"default config: {'PRESENT' if default_config.exists() else 'ABSENT'} ({default_config})")
    print("FINAL: PASS_STATIC_AND_DEMONSTRATION_DATA_CONTRACT; RUNTIME_GATE_PENDING")


if __name__ == "__main__":
    main()
