"""Tiny deterministic adapters for testing the evaluator without simulator/model imports."""
from __future__ import annotations

import numpy as np

from .adapters import CanonicalObservation
from .schema import ActionSpec, ObservationSpec


class MockPolicy:
    name = "mock"
    action_spec = ActionSpec(dim=2)

    def reset(self) -> None:
        pass

    def predict(self, observation: CanonicalObservation) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)


class MockEnvironment:
    observation_spec = ObservationSpec(state_dim=1)
    action_spec = ActionSpec(dim=2)

    def __init__(self, success_after: int = 2) -> None:
        self.success_after = success_after
        self.steps = 0
        self.closed = False

    def reset(self, init_state_index: int, seed: int) -> CanonicalObservation:
        self.steps = 0
        return CanonicalObservation(images={}, state=np.zeros(1, dtype=np.float32), instruction="mock")

    def step(self, action: np.ndarray):
        self.steps += 1
        success = self.steps >= self.success_after
        observation = CanonicalObservation(images={}, state=np.zeros(1, dtype=np.float32), instruction="mock")
        return observation, 1.0 if success else 0.0, success, False, {"is_success": success}

    def close(self) -> None:
        self.closed = True
