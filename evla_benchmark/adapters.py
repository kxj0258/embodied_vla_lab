"""Adapter boundary between policies/environments and the benchmark core."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .schema import ActionSpec, ObservationSpec


@dataclass(frozen=True)
class CanonicalObservation:
    images: dict[str, np.ndarray]
    state: np.ndarray | None
    instruction: str | None
    timestamp_s: float | None = None


class PolicyAdapter(Protocol):
    name: str
    action_spec: ActionSpec

    def reset(self) -> None: ...

    def predict(self, observation: CanonicalObservation) -> np.ndarray: ...


class EnvironmentAdapter(Protocol):
    observation_spec: ObservationSpec
    action_spec: ActionSpec

    def reset(self, init_state_index: int, seed: int) -> CanonicalObservation: ...

    def step(self, action: np.ndarray) -> tuple[CanonicalObservation, float, bool, bool, dict[str, Any]]: ...

    def close(self) -> None: ...
