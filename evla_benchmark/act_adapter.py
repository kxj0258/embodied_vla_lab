"""ACT policy adapter for the pinned LeRobot/LIBERO runtime."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .adapters import CanonicalObservation
from .schema import ActionSpec


class ACTLiberoPolicy:
    name = "act"
    action_spec = ActionSpec(dim=7, chunk_size=20, action_steps_per_inference=5, control_frequency_hz=20)

    def __init__(self, checkpoint: Path, stats: dict[str, np.ndarray], device: str = "cuda") -> None:
        import torch
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        self.torch = torch
        self.device = torch.device(device)
        config = ACTConfig(
            input_features={
                "observation.images.agentview": PolicyFeature(FeatureType.VISUAL, (3, 128, 128)),
                "observation.images.eye_in_hand": PolicyFeature(FeatureType.VISUAL, (3, 128, 128)),
                "observation.state": PolicyFeature(FeatureType.STATE, (9,)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
            chunk_size=20, n_action_steps=5, use_vae=False,
            pretrained_backbone_weights=None, device=device,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.stats = {key: np.asarray(value, dtype=np.float32) for key, value in stats.items()}
        self.clipped_action_count = 0
        self.policy = ACTPolicy(config).to(self.device).eval()
        self.policy.load_state_dict(payload["model"], strict=True)

    def reset(self) -> None:
        self.policy.reset()

    def predict(self, observation: CanonicalObservation) -> np.ndarray:
        images: dict[str, Any] = {}
        for source in ("agentview", "eye_in_hand"):
            image = np.asarray(observation.images[source])
            if image.shape != (128, 128, 3) or image.dtype != np.uint8:
                raise ValueError(f"invalid {source} image contract: {image.shape}/{image.dtype}")
            tensor = self.torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
            images[f"observation.images.{source}"] = self.torch.flip(tensor, dims=(1, 2)).unsqueeze(0).to(self.device)
        if observation.state is None or observation.state.shape != (9,):
            raise ValueError("ACT expects a 9D state")
        images["observation.state"] = self.torch.from_numpy((observation.state - self.stats["sm"]) / self.stats["ss"]).unsqueeze(0).to(self.device)
        normalized = self.policy.select_action(images)
        self.torch.cuda.synchronize() if self.device.type == "cuda" else None
        action = (normalized[0].float().detach().cpu().numpy() * self.stats["as"] + self.stats["am"]).astype(np.float32)
        # Match the frozen Phase 2B execution contract: ACT's denormalized
        # prediction is bounded before it reaches LIBERO.  Keep the count so
        # the adapter can expose saturation diagnostics later.
        clipped = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.clipped_action_count += int(np.any(clipped != action))
        return clipped
