#!/usr/bin/env python3
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/phase6d_feagine_a03_limit_diagnostic_v2_manifest.yaml"


def main() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    checks = []
    for item in manifest["artifacts"]:
        path = ROOT / item["path"]
        data = path.read_bytes()
        checks.append(path.is_file())
        checks.append(len(data) == item["bytes"])
        checks.append(hashlib.sha256(data).hexdigest() == item["sha256"])
    result = yaml.safe_load((ROOT / "results/phase6d_feagine_a03_limit_diagnostic_v2.yaml").read_text())
    checks.extend([
        result["classification"] == "PASS",
        result["decision"] == "SOFT_LIMIT_PENETRATION_WITHIN_DIAGNOSTIC_THRESHOLD",
        result["steps"] == 200,
        result["violating_joint_count"] == 1,
        result["max_slide_penetration_m"] < result["diagnostic_thresholds"]["slide_material_violation_m"],
        result["max_hinge_penetration_rad"] == 0.0,
        result["model_predictions"] == 0 and result["hardware_commands"] == 0,
    ])
    passed = sum(checks)
    print(yaml.safe_dump({"classification": "PASS" if all(checks) else "FAIL", "checks_passed": passed, "checks_total": len(checks)}, sort_keys=False), end="")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
