#!/usr/bin/env python
"""Download and audit one frozen official OpenVLA snapshot; never loads model tensors."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
sys.dont_write_bytecode = True

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "openvla/openvla-7b-finetuned-libero-goal"
REVISION = "fa5ae1e7509348889295bba8e08621d8b55e9baf"
OUTPUT = ROOT / "outputs/reference/phase3a_openvla_libero_goal_fa5ae1e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing snapshot root: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    started = time.perf_counter()
    snapshot = Path(snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        cache_dir=OUTPUT / "hf_cache",
        local_files_only=False,
    ))
    assert snapshot.name == REVISION

    files = []
    total_bytes = 0
    for path in sorted(snapshot.iterdir()):
        if not path.is_file():
            continue
        size = path.stat().st_size
        row = {"path": path.name, "bytes": size, "sha256": sha256(path)}
        files.append(row)
        total_bytes += size
        print(json.dumps(row), flush=True)

    names = {row["path"] for row in files}
    required = {
        "config.json", "dataset_statistics.json", "generation_config.json", "model.safetensors.index.json",
        "preprocessor_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
        "model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors",
    }
    assert required <= names, sorted(required - names)
    config = json.loads((snapshot / "config.json").read_text())
    statistics = json.loads((snapshot / "dataset_statistics.json").read_text())
    serialized = json.dumps({"config": config, "statistics": statistics})
    assert "libero_goal" in serialized
    assert "q01" in serialized and "q99" in serialized

    manifest = {
        "classification": "PASS",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "snapshot": str(snapshot),
        "files": files,
        "file_count": len(files),
        "total_logical_bytes": total_bytes,
        "libero_goal_statistics_present": True,
        "q01_q99_present": True,
        "elapsed_s": time.perf_counter() - started,
        "model_loaded": False,
        "gpu_initialized": False,
    }
    (OUTPUT / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
