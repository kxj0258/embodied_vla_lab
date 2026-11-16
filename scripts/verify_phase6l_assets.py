#!/usr/bin/env python3
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/phase6l_feagine_a03_gripper_margin_scan_v1_manifest.yaml"


def main():
    manifest = yaml.safe_load(MANIFEST.read_text())
    checks = []
    for artifact in manifest["artifacts"]:
        payload = (ROOT / artifact["path"]).read_bytes()
        checks.extend(
            [
                len(payload) == artifact["bytes"],
                hashlib.sha256(payload).hexdigest() == artifact["sha256"],
            ]
        )

    result = yaml.safe_load(
        (ROOT / "results/phase6l_feagine_a03_gripper_margin_scan_v1.yaml").read_text()
    )
    commands = result["command_results"]
    x_penetrations = [
        max(command["episodes"][0]["max_slide_penetration_m"], command["episodes"][1]["max_slide_penetration_m"])
        for command in commands
    ]
    checks.extend(
        [
            manifest["classification"] == "PASS",
            manifest["decision"] == "NO_SAFE_GRIPPER_MARGIN_SELECTED",
            result["classification"] == "PASS" and result["execution_pass"] is True,
            result["decision"] == "NO_SAFE_GRIPPER_MARGIN_SELECTED",
            result["selected_gripper_command"] is None,
            [entry["gripper_command"] for entry in commands] == [0.0, 0.01, 0.02, 0.05],
            all(entry["successes"] == 2 and not entry["strict_pass"] for entry in commands),
            all(x_penetrations[index] > x_penetrations[index + 1] for index in range(3)),
            x_penetrations[-1] > 0.0001,
            all(
                episode["classification"] == "PASS"
                for command in commands
                for episode in command["episodes"][2:]
            ),
            result["new_targets_run"] == 12 and result["total_steps"] == 4800,
            result["cuda_initialized"] is False,
            result["model_predictions"] == 0 and result["hardware_commands"] == 0,
            result["reward"] is None,
        ]
    )
    output = {
        "classification": "PASS" if all(checks) else "FAIL",
        "checks_passed": sum(checks),
        "checks_total": len(checks),
    }
    print(yaml.safe_dump(output, sort_keys=False), end="")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
