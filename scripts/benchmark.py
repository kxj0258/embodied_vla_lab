#!/usr/bin/env python3
"""Unified benchmark entry point for canonical summaries and legacy evidence.

Examples:
  python scripts/benchmark.py import results/phase2b_act_libero_ckpt100000_final_heldout45.yaml
  python scripts/benchmark.py import results/phase3a_openvla_libero_final_heldout45_v1.yaml --json
  python scripts/benchmark.py compare --result act.json openvla.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evla_benchmark.legacy import load_legacy  # noqa: E402
from evla_benchmark.schema import validate_result  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="normalize an existing YAML/JSON result")
    imp.add_argument("path", type=Path)
    imp.add_argument("--policy", default=None)
    imp.add_argument("--json", action="store_true", dest="as_json")
    cmp = sub.add_parser("compare", help="compare normalized result files")
    cmp.add_argument("--result", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "import":
        result = load_legacy(args.path, args.policy)
        errors = validate_result(result)
        payload = result.to_dict()
        payload["validation_errors"] = errors
        if args.as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"{result.policy_name}: {result.task.suite}/{result.task.name}")
            print(f"  {result.success_count}/{result.success_total} success ({result.success_rate:.1%})")
            print(f"  split={result.task.split} states={list(result.task.init_state_indices)}")
            print(f"  failures={result.failures}")
            print(f"  validation={'PASS' if not errors else 'FAIL: ' + '; '.join(errors)}")
        return 0 if not errors else 2
    results = [load_legacy(path) for path in args.result]
    print("policy,task,split,success_rate,success_count,success_total,source")
    for result in results:
        print(f"{result.policy_name},{result.task.suite}/{result.task.name},{result.task.split},{result.success_rate:.6f},{result.success_count},{result.success_total},{result.provenance['source_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
