#!/usr/bin/env python3
"""One-shot, audited pruning of regenerable training payloads.

The selection is intentionally explicit. Results, raw logs, presentation assets,
failed-run directories, canonical inference checkpoints and adapters are excluded.
"""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/storage_prune_recommended_20260904_manifest.yaml"
TOKEN = "RECOMMENDED_PRUNE_20260904"

DIFFUSION = Path("outputs/train/phase1c_diffusion_pusht_baseline_30k_bf16/checkpoints")
PUSHT_DET = Path("outputs/train/phase1b_act_pusht_deterministic_30k_bf16/checkpoints")
PUSHT_FORMAL = Path("outputs/train/phase1b_act_pusht_formal_100k_bf16/checkpoints")
PUSHT_OVERFIT = Path("outputs/train/phase1b_act_pusht_overfit_ep51_bf16/checkpoints")
LIBERO_ACT = Path("outputs/train/phase2b_act_libero_turn_on_stove_deterministic_100k_seed20260826/checkpoints")

DIRECTORIES = [
    DIFFUSION / "010000",
    DIFFUSION / "020000",
    DIFFUSION / "030000/pretrained_model",
    DIFFUSION / "030000/training_state",
    PUSHT_DET / "010000",
    PUSHT_DET / "020000",
    PUSHT_DET / "030000/training_state",
    PUSHT_FORMAL / "025000",
    PUSHT_FORMAL / "050000",
    PUSHT_FORMAL / "075000",
    PUSHT_FORMAL / "100000/training_state",
    PUSHT_OVERFIT / "002000/training_state",
    Path("outputs/build/phase6c_feagine_core_v1"),
]

FILES = [
    Path("outputs/train/phase2b_act_libero_fullsplit_smoke_v3_20step_seed20260826/checkpoints/model_000020.pt"),
    Path("outputs/train/phase2b_act_libero_fullsplit_validation_smoke_v4_20step_seed20260826/checkpoints/model_000020.pt"),
    Path("outputs/train/phase3b_openvla_qlora_rank8_batch1_train20_val10_v3_seed20260827/optimizer_and_rng.pt"),
    Path("outputs/train/phase4c_openvla_wrist_qlora_rank8_train20_val10_v1_seed20260829/optimizer_and_rng.pt"),
]

LIBERO_KEEP = {"model_004460.pt", "model_049060.pt", "model_100000.pt"}

PROTECTED_PREFIXES = (
    "results/raw/",
    "outputs/presentation/",
    "libero/",
    ".git/",
)

REQUIRED_SURVIVORS = [
    DIFFUSION / "030000/pretrained_model_ema/model.safetensors",
    PUSHT_DET / "030000/pretrained_model/model.safetensors",
    PUSHT_FORMAL / "100000/pretrained_model/model.safetensors",
    LIBERO_ACT / "model_004460.pt",
    LIBERO_ACT / "model_049060.pt",
    LIBERO_ACT / "model_100000.pt",
    Path("outputs/train/phase3b_openvla_qlora_rank8_batch1_train20_val10_v3_seed20260827/adapter/adapter_model.safetensors"),
    Path("outputs/train/phase4c_openvla_wrist_qlora_rank8_train20_val10_v1_seed20260829/adapter/adapter_model.safetensors"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_checked(path: Path) -> str:
    resolved = path.resolve(strict=True)
    root = ROOT.resolve(strict=True)
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"target escapes project root: {path}")
    relative = resolved.relative_to(root).as_posix()
    if relative.startswith(PROTECTED_PREFIXES):
        raise RuntimeError(f"target enters protected prefix: {relative}")
    return relative


def collect_targets():
    targets = []
    for relative in DIRECTORIES:
        directory = ROOT / relative
        if not directory.is_dir() or directory.is_symlink():
            raise RuntimeError(f"expected real directory missing: {relative}")
        targets.extend(path for path in directory.rglob("*") if path.is_file())
    libero_dir = ROOT / LIBERO_ACT
    actual = {path.name for path in libero_dir.glob("model_*.pt")}
    if not LIBERO_KEEP <= actual:
        raise RuntimeError("one or more protected LIBERO ACT checkpoints are missing")
    targets.extend(path for path in libero_dir.glob("model_*.pt") if path.name not in LIBERO_KEEP)
    for relative in FILES:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"expected real file missing: {relative}")
        targets.append(path)
    unique = sorted(set(targets))
    for path in unique:
        relative_checked(path)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", metavar="CONFIRMATION_TOKEN")
    args = parser.parse_args()
    if args.execute != TOKEN:
        print(f"dry-run only; pass --execute {TOKEN} to prune")
    if MANIFEST.exists():
        raise RuntimeError(f"refusing to overwrite manifest: {MANIFEST.relative_to(ROOT)}")
    for relative in REQUIRED_SURVIVORS:
        if not (ROOT / relative).is_file():
            raise RuntimeError(f"required survivor missing before prune: {relative}")

    targets = collect_targets()
    records = []
    total = 0
    for index, path in enumerate(targets, 1):
        size = path.stat().st_size
        total += size
        records.append({"path": relative_checked(path), "bytes": size, "sha256": sha256(path)})
        if index % 100 == 0 or index == len(targets):
            print(f"hashed {index}/{len(targets)} files", flush=True)

    preview = {
        "classification": "DRY_RUN",
        "files_selected": len(records),
        "bytes_selected": total,
        "gib_selected": round(total / 1024**3, 6),
    }
    if args.execute != TOKEN:
        print(yaml.safe_dump(preview, sort_keys=False), end="")
        return

    started = time.monotonic()
    manifest = {
        "manifest": "storage_prune_recommended_20260904",
        "classification": "PREDELETE_INVENTORY",
        "operation": "AUTHORIZED_IRREVERSIBLE_LOCAL_PRUNE",
        "selection": "training-resume-state-and-noncanonical-periodic-checkpoints",
        "files_deleted": len(records),
        "bytes_deleted_logical": total,
        "gib_deleted_logical": round(total / 1024**3, 6),
        "protected_survivors": [path.as_posix() for path in REQUIRED_SURVIVORS],
        "preserved_boundaries": [
            "all results and raw logs",
            "all presentation assets",
            "all failed-run directories",
            "Phase 3B v2 protocol-collision directory",
            "invalid LIBERO duplicate evidence",
        ],
        "deleted_files": records,
    }
    # Persist the only recovery metadata before removing payloads.
    MANIFEST.write_text(yaml.safe_dump(manifest, sort_keys=False, width=120))
    for path in targets:
        path.unlink()
    # Remove only now-empty directories under the explicitly selected roots.
    for relative in sorted(DIRECTORIES, key=lambda item: len(item.parts), reverse=True):
        directory = ROOT / relative
        if directory.exists():
            shutil.rmtree(directory)
    for relative in REQUIRED_SURVIVORS:
        if not (ROOT / relative).is_file():
            raise RuntimeError(f"required survivor lost after prune: {relative}")
    remaining = [record["path"] for record in records if (ROOT / record["path"]).exists()]
    if remaining:
        raise RuntimeError(f"selected files remain after prune: {remaining[:3]}")
    manifest["classification"] = "PASS"
    manifest["postdelete_verification"] = {
        "all_selected_files_absent": True,
        "all_required_survivors_present": True,
    }
    MANIFEST.write_text(yaml.safe_dump(manifest, sort_keys=False, width=120))
    print(
        yaml.safe_dump(
            {
                "classification": "PASS",
                "files_deleted": len(records),
                "bytes_deleted_logical": total,
                "gib_deleted_logical": round(total / 1024**3, 6),
                "elapsed_s": time.monotonic() - started,
                "manifest": MANIFEST.relative_to(ROOT).as_posix(),
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"REFUSING_PRUNE: {error}", file=sys.stderr)
        raise SystemExit(2)
