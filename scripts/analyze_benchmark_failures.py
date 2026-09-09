#!/usr/bin/env python3
"""Create per-task and per-state failure analysis from unified results."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "episodes" not in data:
        raise ValueError(f"missing episodes: {path}")
    return data


def summarize(episodes: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for episode in episodes:
        key = (episode.get("task_id"), episode.get("task_name"))
        groups[key].append(episode)
    rows = []
    for (task_id, task_name), items in sorted(groups.items(), key=lambda x: (x[0][0] is None, x[0][0])):
        successes = sum(int(x.get("success", False)) for x in items)
        timeouts = sum(int(x.get("truncated", False)) for x in items)
        steps = [int(x["steps"]) for x in items if x.get("steps") is not None]
        rows.append({
            "task_id": task_id, "task_name": task_name, "success": successes, "total": len(items),
            "success_rate": successes / len(items) if items else 0.0, "timeouts": timeouts,
            "mean_steps": sum(steps) / len(steps) if steps else None,
            "timeout_states": [x.get("init_state_index") for x in items if x.get("truncated", False)],
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default-task-id", type=int, default=None)
    parser.add_argument("--default-task-name", default=None)
    args = parser.parse_args()
    reports = []
    for path in args.result:
        data = load(path)
        if len(data["episodes"]) and all(x.get("task_id") is None for x in data["episodes"]):
            for episode in data["episodes"]:
                episode["task_id"] = args.default_task_id
                episode["task_name"] = args.default_task_name
        reports.append({"source": str(path), "experiment_id": data.get("experiment_id"), "policy": data.get("policy_name"), "overall": {"success": data.get("success_count"), "total": data.get("success_total"), "success_rate": data.get("success_rate"), "failures": data.get("failures")}, "by_task": summarize(data["episodes"])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": "evla.benchmark.failure_analysis.v1", "reports": reports}, indent=2) + "\n")
    md = ["# Benchmark Failure Analysis", "", "This report preserves all failures and does not rerun or select episodes.", ""]
    for report in reports:
        md += [f"## {report['policy']} — `{report['experiment_id']}`", "", f"Overall: **{report['overall']['success']}/{report['overall']['total']}** ({report['overall']['success_rate']:.1%})", "", "| Task | Success | Timeout | Mean steps | Timeout states |", "|---|---:|---:|---:|---|"]
        for item in report["by_task"]:
            name = item["task_name"] or "unknown_task"
            mean_steps = f"{item['mean_steps']:.1f}" if item["mean_steps"] is not None else "-"
            md.append(f"| {item['task_id']}: {name} | {item['success']}/{item['total']} | {item['timeouts']} | {mean_steps} | {item['timeout_states']} |")
        md.append("")
    md += ["## Interpretation", "", "- A timeout is an episode that reached the configured horizon without success.", "- Single-task held-out and multi-task breadth results are reported separately.", "- No failed episode is removed, replaced, or relabeled as success.", ""]
    args.output.with_suffix(".md").write_text("\n".join(md))
    print(args.output)
    print(args.output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
