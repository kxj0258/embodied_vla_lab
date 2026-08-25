#!/usr/bin/env python
"""Verify or explicitly fetch the Phase 1A pinned PushT dataset snapshot.

The default mode is metadata-only and does not create the dataset target. Passing
``--download`` is an explicit write action; the script refuses to merge into an
existing target directory.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset/pusht_v3.yaml"),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly download the pinned snapshot after metadata verification.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return config


def payload_manifest(root: Path) -> dict[str, int]:
    manifest = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        manifest[str(relative)] = path.stat().st_size
    return dict(sorted(manifest.items()))


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    repo_id = config["repo_id"]
    revision = config["revision"]
    target = Path(config["root"])
    expected = config["expected"]

    # The host ALL_PROXY uses a socks:// scheme that httpx 0.28.1 cannot load.
    # Preserve the working HTTP/HTTPS proxies and scope this workaround to this process.
    removed_proxy_keys = []
    for key in ("ALL_PROXY", "all_proxy"):
        if key in os.environ:
            removed_proxy_keys.append(key)
            os.environ.pop(key)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    from huggingface_hub import HfApi, snapshot_download

    print("REMOTE_METADATA")
    print(f"repo id: {repo_id}")
    print(f"requested revision: {revision}")
    print(f"removed process proxy keys: {removed_proxy_keys}")

    info = HfApi().repo_info(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        files_metadata=True,
    )
    siblings = list(info.siblings or [])
    known_sizes = [(item.rfilename, item.size) for item in siblings if item.size is not None]
    missing_sizes = [item.rfilename for item in siblings if item.size is None]
    total_bytes = sum(size for _, size in known_sizes)

    groups = defaultdict(lambda: {"files": 0, "bytes": 0})
    for filename, size in known_sizes:
        top_level = filename.split("/", 1)[0]
        groups[top_level]["files"] += 1
        groups[top_level]["bytes"] += size

    print(f"resolved sha: {info.sha}")
    print(f"file entries: {len(siblings)}")
    print(f"known logical bytes: {total_bytes}")
    print(f"target: {target}")
    print(f"target exists: {target.exists()}")
    for name in sorted(groups):
        print(f"{name}: {groups[name]}")

    assert info.id == repo_id
    assert info.sha == revision
    assert not missing_sizes
    assert len(siblings) == expected["payload_files"]
    assert total_bytes == expected["payload_bytes"]

    if not args.download:
        print("PushT pinned metadata preflight: PASS")
        print("Metadata-only mode; no snapshot download was requested.")
        return

    if target.exists():
        raise SystemExit(
            f"Refusing to merge into existing target: {target}. "
            "Run the offline audit instead."
        )

    resolved_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=target,
    )
    manifest = payload_manifest(target)
    payload_bytes = sum(manifest.values())

    print("\nDOWNLOAD_RESULT")
    print(f"resolved local path: {resolved_path}")
    print(f"payload files: {len(manifest)}")
    print(f"payload bytes: {payload_bytes}")
    for filename, size in manifest.items():
        print(f"{size:>10}  {filename}")

    assert Path(resolved_path).resolve() == target.resolve()
    assert len(manifest) == expected["payload_files"]
    assert payload_bytes == expected["payload_bytes"]
    print("PushT fixed-revision snapshot download: PASS")


if __name__ == "__main__":
    main()
