#!/usr/bin/env python
"""Strictly offline Phase 1A audit for the pinned PushT Dataset v3 snapshot.

This script does not download data or models, instantiate ACT, use CUDA, or train.
It validates the local dataset contract, deterministic split, action padding,
video decoding through the configured backend, and real-stat processors.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml


REQUIRED_PAYLOAD = {
    ".gitattributes",
    "README.md",
    "data/chunk-000/file-000.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "videos/observation.image/chunk-000/file-000.mp4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset/pusht_v3.yaml"),
        help="Dataset audit config relative to the project root.",
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


def chunk_metrics(
    selected_ids: list[int], lengths: dict[int, int], chunk_size: int
) -> dict[str, float | int]:
    total_frames = sum(lengths[episode] for episode in selected_ids)
    total_slots = total_frames * chunk_size
    valid_slots = sum(
        min(chunk_size, lengths[episode] - frame)
        for episode in selected_ids
        for frame in range(lengths[episode])
    )
    frames_with_padding = sum(
        min(chunk_size - 1, lengths[episode]) for episode in selected_ids
    )
    return {
        "frames": total_frames,
        "valid_slots": valid_slots,
        "padded_slots": total_slots - valid_slots,
        "padding_ratio": (total_slots - valid_slots) / total_slots,
        "frames_with_padding": frames_with_padding,
        "frames_with_padding_ratio": frames_with_padding / total_frames,
        "mean_valid_actions": valid_slots / total_frames,
    }


def make_stratified_split(
    episode_ids: list[int],
    lengths: dict[int, int],
    validation_count: int,
    seed: int,
) -> tuple[list[int], list[int], list[tuple[list[int], int]]]:
    ordered = sorted(episode_ids, key=lambda episode: (lengths[episode], episode))
    strata = np.array_split(np.asarray(ordered, dtype=np.int64), validation_count)
    rng = random.Random(seed)

    selections = []
    for stratum in strata:
        stratum_ids = [int(episode) for episode in stratum]
        selections.append((stratum_ids, rng.choice(stratum_ids)))

    validation_ids = sorted(selected for _, selected in selections)
    training_ids = sorted(set(episode_ids) - set(validation_ids))
    return training_ids, validation_ids, selections


def check_payload_and_metadata(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Any]:
    expected = config["expected"]
    manifest = payload_manifest(root)
    payload_bytes = sum(manifest.values())

    print("PAYLOAD")
    print(f"root: {root}")
    print(f"files: {len(manifest)}")
    print(f"bytes: {payload_bytes}")
    for name, size in manifest.items():
        print(f"{size:>10}  {name}")

    assert set(manifest) == REQUIRED_PAYLOAD
    assert len(manifest) == expected["payload_files"]
    assert payload_bytes == expected["payload_bytes"]

    info = json.loads((root / "meta/info.json").read_text())
    stats = json.loads((root / "meta/stats.json").read_text())
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    frames = pq.read_table(root / "data/chunk-000/file-000.parquet")
    tasks = pq.read_table(root / "meta/tasks.parquet")

    print("\nMETADATA")
    print(f"codebase version: {info['codebase_version']}")
    print(f"episodes / frames / tasks: {info['total_episodes']} / {info['total_frames']} / {info['total_tasks']}")
    print(f"fps: {info['fps']}")
    print(f"published splits: {info['splits']}")
    print(f"frame columns: {frames.column_names}")

    assert info["codebase_version"] == config["codebase_version"]
    assert info["fps"] == config["fps"]
    assert info["total_episodes"] == expected["total_episodes"]
    assert info["total_frames"] == expected["total_frames"]
    assert info["total_tasks"] == expected["total_tasks"]
    assert frames.num_rows == expected["total_frames"]
    assert len(episodes) == expected["total_episodes"]
    assert tasks.num_rows == expected["total_tasks"]

    features = info["features"]
    assert features["observation.image"]["shape"] == expected["image_shape_hwc"]
    assert features["observation.state"]["shape"] == expected["state_shape"]
    assert features["action"]["shape"] == expected["action_shape"]
    assert features["observation.image"]["dtype"] == "video"
    assert features["observation.state"]["dtype"] == "float32"
    assert features["action"]["dtype"] == "float32"

    episode_index = np.asarray(frames["episode_index"].combine_chunks())
    frame_index = np.asarray(frames["frame_index"].combine_chunks())
    global_index = np.asarray(frames["index"].combine_chunks())
    timestamp = np.asarray(frames["timestamp"].combine_chunks())
    assert np.array_equal(global_index, np.arange(expected["total_frames"]))

    lengths = {}
    for row in episodes:
        episode = int(row["episode_index"])
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        length = int(row["length"])
        positions = np.flatnonzero(episode_index == episode)
        assert end - start == length
        assert np.array_equal(positions, np.arange(start, end))
        assert np.array_equal(frame_index[positions], np.arange(length))
        assert np.allclose(timestamp[positions], frame_index[positions] / info["fps"], atol=1e-5, rtol=0)
        lengths[episode] = length

    for key in ("observation.image", "observation.state", "action"):
        for stat_name in ("min", "max", "mean", "std", "count"):
            value = np.asarray(stats[key][stat_name])
            assert np.isfinite(value).all()
        assert (np.asarray(stats[key]["std"]) > 0).all()

    return info, stats, episodes, frames


def check_image_sampling(stats: dict[str, Any], lengths: dict[int, int]) -> None:
    from lerobot.datasets.compute_stats import estimate_num_samples

    recomputed = sum(estimate_num_samples(length) for length in lengths.values())
    recorded = int(stats["observation.image"]["count"][0])
    assert recomputed == recorded
    assert all(estimate_num_samples(length) == min(length, 100) for length in lengths.values())
    print("\nIMAGE_STATS_SAMPLING")
    print(f"frame total: {sum(lengths.values())}")
    print(f"sampled image count: {recorded}")


def check_split_and_chunks(config: dict[str, Any], lengths: dict[int, int]) -> tuple[list[int], list[int]]:
    split = config["split"]
    episode_ids = sorted(lengths)
    training_ids, validation_ids, selections = make_stratified_split(
        episode_ids,
        lengths,
        int(split["validation_count"]),
        int(split["seed"]),
    )

    assert validation_ids == split["validation_episodes"]
    assert set(training_ids).isdisjoint(validation_ids)
    assert sorted(training_ids + validation_ids) == episode_ids

    print("\nSPLIT")
    print(f"training episodes: {len(training_ids)}")
    print(f"validation episodes: {validation_ids}")
    print("stratum selections:")
    for index, (stratum_ids, selected) in enumerate(selections):
        stratum_lengths = [lengths[episode] for episode in stratum_ids]
        print(
            f"  {index:02d}: length_range=[{min(stratum_lengths)},{max(stratum_lengths)}] "
            f"selected_episode={selected} selected_length={lengths[selected]}"
        )

    train_lengths = np.asarray([lengths[episode] for episode in training_ids])
    val_lengths = np.asarray([lengths[episode] for episode in validation_ids])
    val_frame_fraction = int(val_lengths.sum()) / sum(lengths.values())
    assert abs(float(train_lengths.mean()) - float(val_lengths.mean())) < 1.0
    assert 0.09 <= val_frame_fraction <= 0.11

    print(f"validation frame fraction: {val_frame_fraction:.4%}")
    print(f"train / validation mean length: {train_lengths.mean():.4f} / {val_lengths.mean():.4f}")
    print("chunk metrics:")
    for chunk_size in config["analysis"]["chunk_candidates"]:
        all_values = chunk_metrics(episode_ids, lengths, int(chunk_size))
        train_values = chunk_metrics(training_ids, lengths, int(chunk_size))
        val_values = chunk_metrics(validation_ids, lengths, int(chunk_size))
        print(
            f"  chunk={chunk_size}: all_padding={all_values['padding_ratio']:.4%} "
            f"train_padding={train_values['padding_ratio']:.4%} "
            f"val_padding={val_values['padding_ratio']:.4%}"
        )
        if int(chunk_size) == config["first_training_candidate"]["chunk_size"]:
            assert abs(float(train_values["padding_ratio"]) - float(val_values["padding_ratio"])) < 0.01

    return training_ids, validation_ids


def check_reader_and_processors(
    config: dict[str, Any],
    root: Path,
    stats: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE
    from lerobot.utils.feature_utils import dataset_to_policy_features

    window_size = int(config["analysis"]["action_window_size"])
    fps = int(config["fps"])
    bounds = {
        int(row["episode_index"]): (
            int(row["dataset_from_index"]),
            int(row["dataset_to_index"]),
        )
        for row in episodes
    }

    dataset = LeRobotDataset(
        repo_id=config["repo_id"],
        root=root,
        revision=config["revision"],
        delta_timestamps={ACTION: [index / fps for index in range(window_size)]},
        video_backend=config["video_backend"],
    )
    assert len(dataset) == config["expected"]["total_frames"]
    assert dataset.revision == config["revision"]
    assert dataset._video_backend == config["video_backend"]

    print("\nDATASET_READER")
    print(f"backend: {dataset._video_backend}")
    for sample_index in config["analysis"]["sample_indices"]:
        sample = dataset[int(sample_index)]
        image = sample[OBS_IMAGE]
        state = sample[OBS_STATE]
        action = sample[ACTION]
        pad = sample["action_is_pad"]
        episode = int(sample["episode_index"])
        absolute_index = int(sample["index"])
        _, episode_end = bounds[episode]
        valid_steps = min(window_size, episode_end - absolute_index)
        expected_pad = window_size - valid_steps

        assert tuple(image.shape) == (3, 96, 96)
        assert image.dtype == torch.float32 and torch.isfinite(image).all()
        assert 0 <= float(image.min()) <= float(image.max()) <= 1
        assert tuple(state.shape) == (2,) and state.dtype == torch.float32
        assert tuple(action.shape) == (window_size, 2) and action.dtype == torch.float32
        assert tuple(pad.shape) == (window_size,) and pad.dtype == torch.bool
        assert int(pad.sum()) == expected_pad
        print(f"sample={sample_index} episode={episode} padding={int(pad.sum())}")

    features = dataset_to_policy_features(dataset.meta.features)
    policy_config = ACTConfig(
        input_features={key: value for key, value in features.items() if key != ACTION},
        output_features={ACTION: features[ACTION]},
        chunk_size=window_size,
        n_action_steps=int(config["first_training_candidate"]["n_action_steps"]),
        pretrained_backbone_weights=None,
        device="cpu",
    )
    preprocessor, postprocessor = make_act_pre_post_processors(policy_config, dataset.meta.stats)
    sample = dataset[int(config["analysis"]["normalization_sample_index"])]
    raw = {
        OBS_IMAGE: sample[OBS_IMAGE].unsqueeze(0),
        OBS_STATE: sample[OBS_STATE].unsqueeze(0),
        ACTION: sample[ACTION].unsqueeze(0),
        "action_is_pad": sample["action_is_pad"].unsqueeze(0),
    }
    processed = preprocessor(raw)

    def stat_tensor(key: str, name: str, reference: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(stats[key][name], dtype=reference.dtype, device=reference.device)

    expected_image = (raw[OBS_IMAGE] - stat_tensor(OBS_IMAGE, "mean", raw[OBS_IMAGE])) / (
        stat_tensor(OBS_IMAGE, "std", raw[OBS_IMAGE]) + 1e-8
    )
    expected_state = (raw[OBS_STATE] - stat_tensor(OBS_STATE, "mean", raw[OBS_STATE])) / (
        stat_tensor(OBS_STATE, "std", raw[OBS_STATE]) + 1e-8
    )
    expected_action = (raw[ACTION] - stat_tensor(ACTION, "mean", raw[ACTION])) / (
        stat_tensor(ACTION, "std", raw[ACTION]) + 1e-8
    )

    assert torch.allclose(processed[OBS_IMAGE], expected_image, atol=1e-6, rtol=1e-6)
    assert torch.allclose(processed[OBS_STATE], expected_state, atol=1e-6, rtol=1e-6)
    assert torch.allclose(processed[ACTION], expected_action, atol=1e-6, rtol=1e-6)
    assert torch.equal(processed["action_is_pad"], raw["action_is_pad"])

    roundtrip = postprocessor(processed[ACTION])
    error = float((roundtrip - raw[ACTION]).abs().max())
    assert torch.allclose(roundtrip, raw[ACTION], atol=1e-5, rtol=1e-6)
    print("\nPROCESSOR")
    print(f"image normalized range: {float(processed[OBS_IMAGE].min()):.6f}..{float(processed[OBS_IMAGE].max()):.6f}")
    print(f"action roundtrip max abs error: {error:.12g}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    root = Path(config["root"])

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    before = payload_manifest(root)
    _, stats, episodes, _ = check_payload_and_metadata(config, root)
    lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes}
    check_split_and_chunks(config, lengths)

    with tempfile.TemporaryDirectory(prefix="evla-phase1a-audit-", dir="/tmp") as cache_dir:
        os.environ["HF_DATASETS_CACHE"] = cache_dir
        check_image_sampling(stats, lengths)
        check_reader_and_processors(config, root, stats, episodes)

    after = payload_manifest(root)
    assert before == after

    print("\nPHASE 1A PUSHT DATASET AUDIT: PASS")
    print(f"config: {config_path}")
    print(f"revision: {config['revision']}")
    print("Dataset payload remained unchanged.")
    print("No network, model load, CUDA, training, GUI, or upload was used.")


if __name__ == "__main__":
    main()
