#!/usr/bin/env python
"""Run lerobot-eval while rendering every requested episode.

This preserves the complete sequential policy RNG stream so a stochastic Diffusion
Policy trajectory can be reproduced by its original episode ordinal, not merely its
environment seed. All regular lerobot-eval CLI arguments are forwarded unchanged.
"""

from __future__ import annotations

from lerobot.scripts import lerobot_eval


_original_eval_policy_all = lerobot_eval.eval_policy_all


def _eval_policy_all_render_every_episode(*args, **kwargs):
    kwargs["max_episodes_rendered"] = kwargs["n_episodes"]
    return _original_eval_policy_all(*args, **kwargs)


lerobot_eval.eval_policy_all = _eval_policy_all_render_every_episode


if __name__ == "__main__":
    lerobot_eval.main()
