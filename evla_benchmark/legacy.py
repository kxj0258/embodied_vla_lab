"""Importers for already-frozen phase result files.

This module deliberately does not rerun experiments.  It normalizes historical
YAML/JSON evidence into the v1 benchmark summary format.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml

from .schema import ActionSpec, BenchmarkResult, ObservationSpec, TaskSpec, infer_failure


def _task(data: dict[str, Any], episodes: list[dict[str, Any]]) -> TaskSpec:
    protocol = data.get("protocol", {})
    task_text = str(protocol.get("task", data.get("task", "unknown/unknown")))
    suite, _, name = task_text.partition("/")
    indices = protocol.get("init_state_indices") or protocol.get("heldout_init_state_indices")
    if isinstance(indices, str):
        match = re.search(r"(\d+)\s*(?:\.\.|-)\s*(\d+)", indices)
        if match:
            start, end = (int(x) for x in match.groups())
            indices = list(range(start, end + 1))
        else:
            indices = [int(x) for x in re.findall(r"\d+", indices)]
    indices = list(indices or [int(x.get("init_state_index", i)) for i, x in enumerate(episodes)])
    return TaskSpec(
        suite=suite, task_id=int(protocol.get("task_id", 0)), name=name or task_text,
        instruction=str(protocol.get("instruction", "")),
        observation=ObservationSpec(image_keys=("agentview", "eye_in_hand"), image_shape=(128, 128, 3), state_dim=9, language_required="openvla" in str(data.get("experiment", "")).lower()),
        action=ActionSpec(dim=7, control_frequency_hz=float(protocol.get("control_frequency_hz", 20)), chunk_size=int(protocol.get("action_chunk", 1)), action_steps_per_inference=int(protocol.get("action_steps_per_inference", 1))),
        episode_horizon=int(protocol.get("episode_horizon", 600)), split="heldout" if "heldout" in str(protocol.get("kind", "")) or "heldout" in str(data.get("experiment", "")) else "development", init_state_indices=tuple(indices),
    )


def load_legacy(path: Path, policy_name: str | None = None) -> BenchmarkResult:
    data = yaml.safe_load(path.read_text()) if path.suffix in {".yaml", ".yml"} else __import__("json").loads(path.read_text())
    episodes = list(data.get("episodes", []))
    if not episodes:
        aggregate = data.get("aggregate", data.get("metrics", {}))
        total = int(data.get("success_total", aggregate.get("success_total", 0)))
        count = int(data.get("success_count", aggregate.get("success_count", 0)))
        episodes = [{"init_state_index": i, "success": i < count} for i in range(total)]
    task = _task(data, episodes)
    failures: dict[str, int] = {}
    for row in episodes:
        category = infer_failure(row).value
        failures[category] = failures.get(category, 0) + 1
    aggregate = data.get("aggregate", data.get("metrics", {}))
    count = int(data.get("success_count", aggregate.get("success_count", sum(bool(x.get("success")) for x in episodes))))
    total = int(data.get("success_total", aggregate.get("success_total", len(episodes))))
    return BenchmarkResult(
        experiment_id=str(data.get("experiment", path.stem)), policy_name=policy_name or ("openvla" if "openvla" in path.name else "act"), task=task,
        classification=str(data.get("classification", "IMPORTED")), success_count=count, success_total=total, episodes=episodes,
        metrics={"source": str(path), **aggregate, **data.get("latency", {})}, provenance={"source_file": str(path), "legacy_import": True}, failures=failures,
    )
