#!/usr/bin/env python3
"""Build a compact, read-only report from unified benchmark result JSON files."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    required = {"experiment_id", "policy_name", "success_count", "success_total", "success_rate", "episodes", "metrics", "failures"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    if int(data["success_count"]) > int(data["success_total"]):
        raise ValueError(f"{path}: invalid success counts")
    return data


def row(data: dict, source: Path) -> dict:
    episodes = data["episodes"]
    steps = [int(x["steps"]) for x in episodes if x.get("steps") is not None]
    inference = [float(x["inference_s"]) for x in episodes if x.get("inference_s") is not None]
    return {
        "policy": data["policy_name"], "experiment_id": data["experiment_id"], "source": str(source),
        "success_count": int(data["success_count"]), "success_total": int(data["success_total"]),
        "success_rate": float(data["success_rate"]), "total_environment_steps": int(data["metrics"].get("total_environment_steps", sum(steps))),
        "mean_steps_per_episode": (sum(steps) / len(steps)) if steps else None,
        "mean_episode_inference_s": (sum(inference) / len(inference)) if inference else None,
        "mean_episode_wall_s": float(data["metrics"].get("mean_episode_wall_s", 0.0)),
        "failures": json.dumps(data["failures"], sort_keys=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmark/report_v1"))
    args = parser.parse_args()
    rows = [row(load(path), path) for path in args.result]
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"refusing to overwrite report directory: {output}")
    output.mkdir(parents=True)
    fields = list(rows[0]) if rows else []
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    payload = {"schema_version": "evla.benchmark.report.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "results": rows}
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Unified VLA Benchmark Report", "", f"Generated: `{payload['generated_at_utc']}`", "", "> These are development results from the unified evaluator; they do not replace the preregistered held-out final results.", "", "| Policy | Success | Steps | Mean inference / episode | Mean wall / episode | Failures |", "|---|---:|---:|---:|---:|---|"]
    for item in rows:
        lines.append(f"| {item['policy']} | {item['success_count']}/{item['success_total']} ({item['success_rate']:.1%}) | {item['total_environment_steps']} | {item['mean_episode_inference_s']:.3f}s | {item['mean_episode_wall_s']:.3f}s | `{item['failures']}` |")
    lines += ["", "## Interpretation", "", "- ACT and OpenVLA are evaluated on the same five development init states.", "- OpenVLA inference is isolated in a persistent worker; ACT runs in the LeRobot environment process.", "- Success rates are not claims about unseen tasks, perturbations, or the 45-state held-out final protocol.", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(f"wrote {output / 'report.md'}")
    print(f"wrote {output / 'summary.csv'}")
    print(f"wrote {output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
