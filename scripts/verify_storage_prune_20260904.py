#!/usr/bin/env python3
import hashlib
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/storage_prune_recommended_20260904_manifest.yaml"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    manifest = yaml.safe_load(MANIFEST.read_text())
    deleted = manifest["deleted_files"]
    survivors = [ROOT / path for path in manifest["protected_survivors"]]
    checks = {
        "manifest_pass": manifest["classification"] == "PASS",
        "authorized_operation_recorded": manifest["operation"] == "AUTHORIZED_IRREVERSIBLE_LOCAL_PRUNE",
        "deleted_count": len(deleted) == manifest["files_deleted"] == 550,
        "deleted_bytes": sum(item["bytes"] for item in deleted) == manifest["bytes_deleted_logical"],
        "all_deleted_payloads_absent": all(not (ROOT / item["path"]).exists() for item in deleted),
        "all_required_survivors_present": all(path.is_file() for path in survivors),
        "raw_logs_preserved": (ROOT / "results/raw").is_dir(),
        "presentation_preserved": (ROOT / "outputs/presentation").is_dir(),
        "phase3b_v2_failure_boundary_preserved": (ROOT / "outputs/train/phase3b_openvla_qlora_rank8_batch1_train20_val10_v2_seed20260827").is_dir(),
        "invalid_libero_duplicate_preserved": (ROOT / "libero/libero/datasets/libero_goal/turn_on_the_stove_demo.hdf5.invalid-resume-duplicate").is_file(),
    }
    survivor_hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in survivors}
    classification = "PASS" if all(checks.values()) else "FAIL"
    print(yaml.safe_dump({"classification": classification, "checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks), "survivor_sha256": survivor_hashes}, sort_keys=False), end="")
    raise SystemExit(0 if classification == "PASS" else 1)

if __name__ == "__main__":
    main()
