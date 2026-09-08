"""Framework-independent closed-loop evaluator."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .adapters import EnvironmentAdapter, PolicyAdapter
from .schema import BenchmarkResult, FailureCategory, TaskSpec, infer_failure


@dataclass(frozen=True)
class EvaluatorConfig:
    experiment_id: str
    policy_name: str
    seed: int
    episode_horizon: int
    action_low: float = -1.0
    action_high: float = 1.0
    strict_bounds: bool = True


class ActionValidationError(ValueError):
    pass


def validate_action(action: Any, dim: int, low: float, high: float, strict: bool = True) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.shape != (dim,):
        raise ActionValidationError(f"expected action shape {(dim,)}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ActionValidationError("action contains NaN or Inf")
    if strict and (np.any(array < low) or np.any(array > high)):
        raise ActionValidationError(f"action outside [{low}, {high}]")
    return np.clip(array, low, high).astype(np.float32)


def _failure(row: dict[str, Any]) -> str:
    return infer_failure(row).value


def evaluate(
    policy: PolicyAdapter,
    environment: EnvironmentAdapter,
    task: TaskSpec,
    config: EvaluatorConfig,
) -> BenchmarkResult:
    if policy.action_spec.dim != task.action.dim or environment.action_spec.dim != task.action.dim:
        raise ValueError("policy, environment and task action dimensions must agree")
    episodes: list[dict[str, Any]] = []
    for episode, init_index in enumerate(task.init_state_indices):
        row: dict[str, Any] = {"episode": episode, "init_state_index": init_index, "seed": config.seed + episode, "success": False}
        started = time.perf_counter()
        inference_s = 0.0
        try:
            observation = environment.reset(init_index, config.seed + episode)
            policy.reset()
            for step in range(1, config.episode_horizon + 1):
                infer_started = time.perf_counter()
                action = policy.predict(observation)
                inference_s += time.perf_counter() - infer_started
                action = validate_action(action, task.action.dim, config.action_low, config.action_high, config.strict_bounds)
                observation, reward, terminated, truncated, info = environment.step(action)
                row["steps"] = step
                row["sum_reward"] = float(row.get("sum_reward", 0.0)) + float(reward)
                row["max_reward"] = max(float(row.get("max_reward", -math.inf)), float(reward))
                row["success"] = bool(row["success"] or info.get("is_success", False))
                row["terminated"] = bool(terminated)
                row["truncated"] = bool(truncated)
                if terminated or truncated or row["success"]:
                    break
            else:
                row["truncated"] = True
        except ActionValidationError as exc:
            row.update({"error": str(exc), "failure_category": FailureCategory.INVALID_ACTION.value})
        except TimeoutError as exc:
            row.update({"error": str(exc), "failure_category": FailureCategory.INFERENCE_TIMEOUT.value})
        except (FloatingPointError, ValueError) as exc:
            row.update({"error": str(exc), "failure_category": FailureCategory.NUMERICAL_ERROR.value})
        except Exception as exc:  # preserve episode-level evidence and rethrow only after cleanup
            row.update({"error": f"{type(exc).__name__}: {exc}", "failure_category": FailureCategory.ENVIRONMENT_ERROR.value})
        finally:
            row["inference_s"] = inference_s
            row["episode_wall_s"] = time.perf_counter() - started
            episodes.append(row)
    environment.close()
    failures: dict[str, int] = {}
    for row in episodes:
        category = row.get("failure_category", _failure(row))
        failures[category] = failures.get(category, 0) + 1
    success_count = sum(int(row["success"]) for row in episodes)
    return BenchmarkResult(
        experiment_id=config.experiment_id, policy_name=config.policy_name, task=task,
        classification="PASS", success_count=success_count, success_total=len(episodes),
        episodes=episodes,
        metrics={
            "mean_inference_s": float(np.mean([row["inference_s"] for row in episodes])) if episodes else 0.0,
            "mean_episode_wall_s": float(np.mean([row["episode_wall_s"] for row in episodes])) if episodes else 0.0,
            "total_environment_steps": int(sum(int(row.get("steps", 0)) for row in episodes)),
        }, failures=failures,
        provenance={"evaluator": "evla_benchmark.evaluator", "schema_version": "evla.benchmark.v1"},
    )
