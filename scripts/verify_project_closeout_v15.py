#!/usr/bin/env python3
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/project_closeout_v15_phase6l_v1_manifest.yaml"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = yaml.safe_load(MANIFEST.read_text())
    checks = [
        manifest["classification"] == "PASS",
        manifest["generation_mode"] == "cpu_only_from_frozen_evidence",
        manifest["environment_steps"] == 0,
    ]
    for artifact in manifest["documents"] + manifest["canonical_evidence"]:
        path = ROOT / artifact["path"]
        checks.extend(
            [
                path.is_file(),
                path.stat().st_size == artifact["bytes"],
                digest(path) == artifact["sha256"],
            ]
        )
    checks.append((ROOT / manifest["historical_evidence"]).is_file())
    output = {
        "classification": "PASS" if all(checks) else "FAIL",
        "checks_passed": sum(checks),
        "checks_total": len(checks),
        "environment_steps": 0,
    }
    print(yaml.safe_dump(output, sort_keys=False), end="")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
