"""Small, dependency-light benchmark contracts for Embodied VLA Lab."""

from .schema import (
    ActionSpec,
    BenchmarkResult,
    FailureCategory,
    ObservationSpec,
    TaskSpec,
    validate_result,
)

__all__ = [
    "ActionSpec",
    "BenchmarkResult",
    "FailureCategory",
    "ObservationSpec",
    "TaskSpec",
    "validate_result",
]
