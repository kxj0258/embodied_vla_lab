# P0 MVP presentation and reproducibility closure

Date: 2026-08-27  
Status: `COMPLETE / PASS`

## Purpose and evidence boundary

P0 packages the completed Phase 0–3B evidence without changing any frozen experiment. Offline composition performs
zero environment steps and zero model inference. It does not create new policy evidence, revise checkpoint selection,
or convert offline loss/token accuracy into reward or success.

Canonical structured manifest: `results/p0_mvp_v2_manifest.yaml`; local media are under
`outputs/presentation/p0_mvp_v2_20260827`. The preserved v1 build passed
generation but failed visual QA because 128×128 observations were not enlarged and comparison headings overlapped;
it remains immutable build history and is not the canonical presentation candidate.

## Delivered assets

- OpenVLA frozen-base held-out init5 strict-success video, 83 frames/steps.
- Base versus QLoRA typical speedup at held-out init40: base 221 steps, adapter 70, both successful.
- Base versus QLoRA init44 tail comparison: base succeeds at 68; adapter reaches the frozen 600-step cap and fails.
- Paired 889-frame offline metric figure, explicitly labeled as not environment success.
- Frozen 45-state reliability figure: ACT 45/45, OpenVLA base 45/45, QLoRA 43/45.
- Dependency-isolated system architecture diagram in SVG and PNG.
- SHA-256/ffprobe manifest and read-only verifier.

All videos are 1280×720 H.264 at 10 presentation fps. They replay frozen PNG observations with factual overlays;
presentation fps is not control frequency or measured inference throughput.

## ACT capture boundary

The Phase 2 final preserved metrics but no observation frames. Therefore a real ACT policy video cannot be derived
honestly from existing files. `scripts/record_p0_act_libero_demo.py` and
`configs/eval/act_libero_p0_demo_init05_v1.yaml` implement exactly one headless recapture of the already-frozen
step-100k checkpoint and held-out init5. The new ID/output/log/result paths refuse overwrite. This run is
presentation-only and cannot revise the prior 45/45 final.

The host capture PASSed at 83 steps with strict success and reward 1.0. It used 17 policy forwards, 0.575 s total
timed inference, 162,552,832/216,006,656-byte peak allocated/reserved VRAM, 21.44 s host wall and 3,826,808 KiB
maximum RSS. No training/offline validation/gradient/optimizer/checkpoint creation metric applies. The capture-v1
title was clipped during visual QA, so it remains immutable but noncanonical; a zero-environment/zero-inference
offline recomposition produced the canonical presentation-v2 video.

## Claims

Safe: one fixed LIBERO task/runtime; complete finite held-out init5–49 enumeration; ACT and base both 45/45;
QLoRA paired offline gains but held-out reliability regression to 43/45. Unsafe: general LIBERO 100%, language or
viewpoint generalization, real-time 20 Hz OpenVLA, real-world transfer, or QLoRA superiority.

## Verification

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_p0_assets.py
```
