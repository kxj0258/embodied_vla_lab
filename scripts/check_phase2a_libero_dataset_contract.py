#!/usr/bin/env python
"""Read-only Phase 2A LIBERO split, statistics, and ACT sample contract."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dataset/libero_goal_turn_on_stove_phase2a.yaml"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    path = ROOT / cfg["dataset"]["path"]
    assert digest(path) == cfg["dataset"]["sha256"]
    train_ids = cfg["split"]["train_episodes"]
    val_ids = cfg["split"]["validation_episodes"]
    assert len(train_ids) == 40 and len(val_ids) == 10
    assert not set(train_ids) & set(val_ids)
    assert sorted(train_ids + val_ids) == list(range(50))
    ids = np.arange(50); np.random.default_rng(cfg["split"]["seed"]).shuffle(ids)
    assert sorted(ids[:40].tolist()) == train_ids and sorted(ids[40:].tolist()) == val_ids

    with h5py.File(path, "r") as file:
        data = file["data"]
        def arrays(episode_ids):
            actions=[]; states=[]; frames=0
            for episode in episode_ids:
                group=data[f"demo_{episode}"]
                length=len(group["actions"]); frames += length
                assert group["obs/agentview_rgb"].shape == (length,128,128,3)
                assert group["obs/eye_in_hand_rgb"].shape == (length,128,128,3)
                actions.append(group["actions"][()])
                states.append(np.concatenate([group["obs/joint_states"][()], group["obs/gripper_states"][()]], axis=1))
            return np.concatenate(actions), np.concatenate(states), frames
        train_action, train_state, train_frames = arrays(train_ids)
        val_action, val_state, val_frames = arrays(val_ids)
        assert train_frames + val_frames == 4460
        assert train_action.shape == (train_frames,7) and train_state.shape == (train_frames,9)
        assert val_action.shape == (val_frames,7) and val_state.shape == (val_frames,9)
        assert np.isfinite(train_action).all() and np.isfinite(train_state).all()

        group=data[f"demo_{train_ids[0]}"]
        index=len(group["actions"])-3; chunk=20
        image=np.moveaxis(group["obs/agentview_rgb"][index],-1,0).astype(np.float32)/255
        state=np.concatenate([group["obs/joint_states"][index],group["obs/gripper_states"][index]]).astype(np.float32)
        action=np.zeros((chunk,7),dtype=np.float32); mask=np.ones(chunk,dtype=bool)
        available=min(chunk,len(group["actions"])-index)
        action[:available]=group["actions"][index:index+available]; mask[:available]=False
        assert image.shape==(3,128,128) and 0 <= image.min() <= image.max() <= 1
        assert state.shape==(9,) and action.shape==(20,7) and mask.shape==(20,)
        assert available==3 and mask.sum()==17

    print("classification: PASS")
    print(f"train_episodes_frames: 40 {train_frames}")
    print(f"validation_episodes_frames: 10 {val_frames}")
    print("state_mean:", train_state.mean(0).tolist())
    print("state_std:", train_state.std(0).tolist())
    print("action_mean:", train_action.mean(0).tolist())
    print("action_std:", train_action.std(0).tolist())
    print("action_min:", train_action.min(0).tolist())
    print("action_max:", train_action.max(0).tolist())
    print("terminal_sample: image [3,128,128], state [9], action [20,7], pad [20] with 17 padded")
    print("FINAL: PASS_PHASE2A_LIBERO_DATASET_PREPROCESSING_CONTRACT")


if __name__ == "__main__":
    main()

