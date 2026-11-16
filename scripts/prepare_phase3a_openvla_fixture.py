#!/usr/bin/env python3
"""Export one frozen, real LIBERO agent-view frame for the OpenVLA load smoke."""

import hashlib
import json
from pathlib import Path

import h5py
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "libero/libero/datasets/libero_goal/turn_on_the_stove_demo.hdf5"
OUTPUT = ROOT / "outputs/reference/phase3a_openvla_fixture_demo0_t000"
IMAGE = OUTPUT / "agentview_rgb.png"
MANIFEST = OUTPUT / "fixture_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite fixture: {OUTPUT}")
    with h5py.File(SOURCE, "r") as dataset:
        frame = dataset["data/demo_0/obs/agentview_rgb"][0]
    if frame.shape != (128, 128, 3) or frame.dtype.name != "uint8":
        raise AssertionError(f"unexpected frame contract: {frame.shape} {frame.dtype}")
    OUTPUT.mkdir(parents=True)
    Image.fromarray(frame, mode="RGB").save(IMAGE, format="PNG", optimize=False)
    result = {
        "classification": "PASS",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "demo": "demo_0",
        "timestep": 0,
        "observation_key": "agentview_rgb",
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "orientation": "stored_demonstration_orientation_no_additional_rotation",
        "image": str(IMAGE.relative_to(ROOT)),
        "image_bytes": IMAGE.stat().st_size,
        "image_sha256": sha256(IMAGE),
    }
    MANIFEST.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
