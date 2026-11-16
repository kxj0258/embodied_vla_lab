#!/usr/bin/env python
"""Offline EGL replay gate for Phase 2A LIBERO demo_0.

Requires explicit environment-execution authorization. It performs no network access,
CUDA policy forward, training, formal evaluation, GUI, video write, or artifact write.
"""

from __future__ import annotations

import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/evla_phase2a_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/evla_phase2a_matplotlib")
sys.dont_write_bytecode = True

import h5py
import numpy as np

from libero.libero.envs import TASK_MAPPING
from libero.libero.utils import utils as libero_utils


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "libero/libero/datasets/libero_goal/turn_on_the_stove_demo.hdf5"
BDDL = ROOT / "libero/libero/libero/bddl_files/libero_goal/turn_on_the_stove.bddl"
EXPECTED_DEMO_SHA256 = "387fc10747696b80dea6ed8d7f2beaa162bf92ae11750b241073cbd33aac73d5"
ASSETS = Path("/home/kxj/.cache/libero/assets")


def validate_image(name: str, image: np.ndarray) -> None:
    assert image.shape == (128, 128, 3), (name, image.shape)
    assert image.dtype == np.uint8, (name, image.dtype)
    assert np.isfinite(image).all(), name
    assert int(image.max()) > int(image.min()), name


def migrate_libero_asset_paths(xml: str) -> str:
    tree = ET.fromstring(xml)
    asset = tree.find("asset")
    assert asset is not None
    replacements = 0
    for element in list(asset.findall("mesh")) + list(asset.findall("texture")):
        old = element.get("file")
        if old is None or "/chiliocosm/assets/" not in old:
            continue
        suffix = old.split("/chiliocosm/assets/", 1)[1]
        target = ASSETS / suffix
        assert target.is_file(), target
        element.set("file", str(target))
        replacements += 1
    migrated = ET.tostring(tree, encoding="unicode")
    assert replacements > 0
    assert "/chiliocosm/assets/" not in migrated
    assert "/Users/" not in migrated
    return migrated


def main() -> None:
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("LIBERO_CONFIG_PATH"), "Set isolated LIBERO_CONFIG_PATH"
    assert not (Path.home() / ".libero/config.yaml").exists()

    start = time.perf_counter()
    with h5py.File(DEMO, "r") as file:
        data = file["data"]
        group = data["demo_0"]
        actions = group["actions"][()]
        states = group["states"][()]
        model_xml = libero_utils.postprocess_model_xml(group.attrs["model_file"], {})
        model_xml = migrate_libero_asset_paths(model_xml)
        metadata = json.loads(data.attrs["env_args"])
        env_kwargs = metadata["env_kwargs"]
        env_kwargs.update(
            bddl_file_name=str(BDDL),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["robot0_eye_in_hand", "agentview"],
            camera_heights=128,
            camera_widths=128,
            camera_depths=False,
            camera_segmentations=None,
            control_freq=20,
            hard_reset=True,
        )

        env = TASK_MAPPING[metadata["problem_name"]](**env_kwargs)
        try:
            env.reset_from_xml_string(model_xml)
            env.sim.reset()
            env.sim.set_state_from_flattened(group.attrs["init_state"])
            env.sim.forward()

            max_next_state_l2 = 0.0
            next_state_errors = []
            qpos_errors = []
            qvel_errors = []
            initial_state_l2 = float(np.linalg.norm(states[0] - env.sim.get_state().flatten()))
            success_step = None
            reward_sum = 0.0
            first_obs = None
            last_obs = None
            for index, action in enumerate(actions):
                obs, reward, done, _ = env.step(action)
                if first_obs is None:
                    first_obs = obs
                last_obs = obs
                reward_sum += float(reward)
                if index < len(actions) - 1:
                    actual = env.sim.get_state().flatten()
                    error = float(np.linalg.norm(states[index + 1] - actual))
                    next_state_errors.append(error)
                    nq = env.sim.model.nq
                    qpos_errors.append(float(np.linalg.norm(states[index + 1][1 : 1 + nq] - actual[1 : 1 + nq])))
                    qvel_errors.append(float(np.linalg.norm(states[index + 1][1 + nq :] - actual[1 + nq :])))
                    max_next_state_l2 = max(max_next_state_l2, error)
                if done and success_step is None:
                    success_step = index + 1

            assert first_obs is not None and last_obs is not None
            for label, obs in [("first", first_obs), ("last", last_obs)]:
                validate_image(f"{label}/agentview", obs["agentview_image"])
                validate_image(f"{label}/eye_in_hand", obs["robot0_eye_in_hand_image"])
                for key in ["robot0_gripper_qpos", "robot0_joint_pos", "robot0_eef_pos", "robot0_eef_quat"]:
                    assert np.isfinite(obs[key]).all(), (label, key)

            final_success = bool(env._check_success())
            elapsed = time.perf_counter() - start
            deterministic_replay_pass = max_next_state_l2 <= 0.01
            classification = "PASS" if deterministic_replay_pass and final_success else "FAIL"
            print(f"classification: {classification}")
            print(f"demo_sha256: {EXPECTED_DEMO_SHA256}")
            print(f"actions_executed: {len(actions)}")
            print(f"action_shape_dtype: {actions.shape} {actions.dtype}")
            print(f"max_next_state_l2: {max_next_state_l2:.12g}")
            print(f"initial_state_l2: {initial_state_l2:.12g}")
            print(f"mean_next_state_l2: {float(np.mean(next_state_errors)):.12g}")
            print(f"median_next_state_l2: {float(np.median(next_state_errors)):.12g}")
            print(f"p95_next_state_l2: {float(np.quantile(next_state_errors, 0.95)):.12g}")
            print(f"max_qpos_l2: {max(qpos_errors):.12g}")
            print(f"median_qpos_l2: {float(np.median(qpos_errors)):.12g}")
            print(f"max_qvel_l2: {max(qvel_errors):.12g}")
            print(f"median_qvel_l2: {float(np.median(qvel_errors)):.12g}")
            print(f"first_step_over_0_01: {next((i + 1 for i, value in enumerate(next_state_errors) if value > 0.01), None)}")
            if success_step is not None:
                print(f"max_next_state_l2_before_success: {max(next_state_errors[: max(success_step - 1, 1)]):.12g}")
            print(f"reward_sum: {reward_sum}")
            print(f"success_step: {success_step}")
            print(f"final_success: {final_success}")
            print(
                "agentview_first_last_range: "
                f"[{int(first_obs['agentview_image'].min())},{int(first_obs['agentview_image'].max())}] "
                f"[{int(last_obs['agentview_image'].min())},{int(last_obs['agentview_image'].max())}]"
            )
            print(
                "eye_in_hand_first_last_range: "
                f"[{int(first_obs['robot0_eye_in_hand_image'].min())},{int(first_obs['robot0_eye_in_hand_image'].max())}] "
                f"[{int(last_obs['robot0_eye_in_hand_image'].min())},{int(last_obs['robot0_eye_in_hand_image'].max())}]"
            )
            print(f"replay_wall_s: {elapsed:.6f}")
            assert deterministic_replay_pass
            assert success_step is not None
            assert final_success
            assert reward_sum >= 1.0
            print("FINAL: PASS_DEMO0_OFFLINE_EGL_REPLAY")
        finally:
            env.close()


if __name__ == "__main__":
    main()
