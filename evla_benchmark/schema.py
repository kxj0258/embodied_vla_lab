"""Canonical, JSON/YAML-friendly schemas used by benchmark tooling.

The schema is intentionally framework independent.  Runtime adapters can use
it without importing LeRobot, LIBERO, torch, or the OpenVLA environment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class FailureCategory(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    TERMINATION_WITHOUT_SUCCESS = "termination_without_success"
    INVALID_ACTION = "invalid_action"
    INFERENCE_TIMEOUT = "inference_timeout"
    NUMERICAL_ERROR = "numerical_error"
    ENVIRONMENT_ERROR = "environment_error"
    PROTOCOL_ERROR = "protocol_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ObservationSpec:
    image_keys: tuple[str, ...] = ()
    image_shape: tuple[int, int, int] | None = None
    image_dtype: str = "uint8"
    state_dim: int | None = None
    language_required: bool = False
    camera_frame: str = "environment"
    state_frame: str = "robot"


@dataclass(frozen=True)
class ActionSpec:
    dim: int
    representation: str = "delta_pose_gripper"
    coordinate_frame: str = "environment"
    rotation_representation: str = "axis_angle"
    gripper_semantics: str = "normalized_open_close"
    normalization: str = "policy_defined"
    control_frequency_hz: float | None = None
    chunk_size: int = 1
    action_steps_per_inference: int = 1


@dataclass(frozen=True)
class TaskSpec:
    suite: str
    task_id: int
    name: str
    instruction: str
    observation: ObservationSpec
    action: ActionSpec
    episode_horizon: int
    split: str
    init_state_indices: tuple[int, ...]


@dataclass
class BenchmarkResult:
    experiment_id: str
    policy_name: str
    task: TaskSpec
    classification: str
    success_count: int
    success_total: int
    episodes: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)
    schema_version: str = "evla.benchmark.v1"

    @property
    def success_rate(self) -> float:
        return self.success_count / self.success_total if self.success_total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"success_rate": self.success_rate}


def _tuple(value: Any) -> tuple:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def task_from_dict(data: Mapping[str, Any]) -> TaskSpec:
    obs = data.get("observation", {})
    act = data.get("action", {})
    return TaskSpec(
        suite=str(data["suite"]), task_id=int(data["task_id"]), name=str(data["name"]),
        instruction=str(data.get("instruction", "")),
        observation=ObservationSpec(
            image_keys=tuple(str(x) for x in obs.get("image_keys", [])),
            image_shape=tuple(obs["image_shape"]) if obs.get("image_shape") else None,
            image_dtype=str(obs.get("image_dtype", "uint8")),
            state_dim=int(obs["state_dim"]) if obs.get("state_dim") is not None else None,
            language_required=bool(obs.get("language_required", False)),
            camera_frame=str(obs.get("camera_frame", "environment")),
            state_frame=str(obs.get("state_frame", "robot")),
        ),
        action=ActionSpec(
            dim=int(act["dim"]), representation=str(act.get("representation", "delta_pose_gripper")),
            coordinate_frame=str(act.get("coordinate_frame", "environment")),
            rotation_representation=str(act.get("rotation_representation", "axis_angle")),
            gripper_semantics=str(act.get("gripper_semantics", "normalized_open_close")),
            normalization=str(act.get("normalization", "policy_defined")),
            control_frequency_hz=float(act["control_frequency_hz"]) if act.get("control_frequency_hz") is not None else None,
            chunk_size=int(act.get("chunk_size", 1)),
            action_steps_per_inference=int(act.get("action_steps_per_inference", 1)),
        ),
        episode_horizon=int(data["episode_horizon"]), split=str(data["split"]),
        init_state_indices=tuple(int(x) for x in data["init_state_indices"]),
    )


def infer_failure(row: Mapping[str, Any]) -> FailureCategory:
    if bool(row.get("success", False)):
        return FailureCategory.SUCCESS
    if bool(row.get("timeout", False)) or bool(row.get("truncated", False)):
        return FailureCategory.TIMEOUT
    if row.get("error"):
        return FailureCategory.ENVIRONMENT_ERROR
    if row.get("terminated", False):
        return FailureCategory.TERMINATION_WITHOUT_SUCCESS
    return FailureCategory.UNKNOWN


def validate_result(result: BenchmarkResult) -> list[str]:
    errors: list[str] = []
    if result.success_count < 0 or result.success_total < 0 or result.success_count > result.success_total:
        errors.append("success_count must satisfy 0 <= success_count <= success_total")
    if result.success_total != len(result.task.init_state_indices):
        errors.append("success_total must equal number of init_state_indices")
    if result.task.action.dim <= 0:
        errors.append("action.dim must be positive")
    for row in result.episodes:
        if "init_state_index" not in row:
            errors.append("episode is missing init_state_index")
        if "success" not in row:
            errors.append("episode is missing success")
    return errors
