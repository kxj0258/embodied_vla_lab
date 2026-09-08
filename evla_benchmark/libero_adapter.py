"""LIBERO environment adapter matching the frozen Phase 2B observation contract."""
from __future__ import annotations

from typing import Any

import numpy as np

from .adapters import CanonicalObservation
from .schema import ActionSpec, ObservationSpec


class LiberoGoalEnvironment:
    observation_spec = ObservationSpec(
        image_keys=("agentview", "eye_in_hand"), image_shape=(128, 128, 3), state_dim=9,
        camera_frame="libero_camera", state_frame="robot0",
    )
    action_spec = ActionSpec(dim=7, coordinate_frame="libero_relative", chunk_size=20, action_steps_per_inference=5, control_frequency_hz=20)

    def __init__(self, suite_name: str, task_id: int, task_order_index: int, episode_horizon: int = 600) -> None:
        from lerobot.envs.libero import LiberoEnv
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=task_order_index)
        self.env = LiberoEnv(
            task_suite=suite, task_id=task_id, task_suite_name=suite_name, episode_length=episode_horizon,
            camera_name=["agentview_image", "robot0_eye_in_hand_image"],
            camera_name_mapping={"agentview_image": "agentview", "robot0_eye_in_hand_image": "eye_in_hand"},
            obs_type="pixels_agent_pos", observation_width=128, observation_height=128,
            init_states=True, episode_index=0, n_envs=1, num_steps_wait=5,
            control_freq=20, control_mode="relative", hard_reset=True,
        )

    @staticmethod
    def _canonical(observation: dict[str, Any]) -> CanonicalObservation:
        pixels = observation["pixels"]
        robot = observation["robot_state"]
        state = np.concatenate([robot["joints"]["pos"], robot["gripper"]["qpos"]]).astype(np.float32)
        return CanonicalObservation(images={key: np.asarray(pixels[key]) for key in ("agentview", "eye_in_hand")}, state=state, instruction=None)

    def reset(self, init_state_index: int, seed: int) -> CanonicalObservation:
        self.env.unwrapped.episode_index = int(init_state_index) if hasattr(self.env, "unwrapped") else int(init_state_index)
        observation, _ = self.env.reset(seed=seed)
        return self._canonical(observation)

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._canonical(observation), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()
