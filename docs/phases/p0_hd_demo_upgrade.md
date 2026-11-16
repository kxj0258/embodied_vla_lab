# P0-HD 640px demo upgrade

Date: 2026-08-28  
Status: `COMPLETE`

## Goal

Replace presentation-only 128px enlargements with genuine 640×640 LIBERO rendering while preserving every frozen
policy input contract and all Phase 2/3 conclusions. The final presentation canvas is 1920×1080; H.264 uses CRF 16,
slow preset, yuv420p and 20 display fps. Each run has a new ID/output/log/result and refuses overwrite.

## Resolution separation

```text
single native-128 LIBERO environment ──→ frozen policy observation and physics step
                │
                └─ same MuJoCo sim + same EGL context, direct 640 camera render
                                                └─→ clean master and 1080p presentation video
```

No post-hoc AI super-resolution is used. The high-resolution view is evidence from the simulator renderer, not
hallucinated detail. These presentation recaptures cannot revise checkpoint selection, held-out success estimates or
the immutable Phase 3B conclusion.

## Planned canonical captures

1. ACT step-100k, held-out init5 success.
2. OpenVLA frozen base, held-out init5 success.
3. Base versus QLoRA typical-speedup state init40.
4. Base versus QLoRA tail state init44, including the frozen adapter failure mode.

The first ACT 640-downsample gate PASSed at 82 steps and proved the visual-quality gain, but it changed source pixels
relative to the native 128 renderer and is therefore noncanonical. The first OpenVLA v1 design used two EGL-backed
environments; it interrupted during environment initialization before the worker/model or any environment step and
left the NVIDIA driver unavailable. That run is frozen as infrastructure `FAIL`, not policy evidence.

The replacement v2 design uses one native-128 environment only. OpenVLA receives the unchanged native 128 image,
while the same MuJoCo simulation and the same offscreen context issue an additional direct 640 camera render. The
v2 script and base-init5 config pass static syntax/schema/no-overwrite inspection. Formal execution is blocked until
the NVIDIA driver is restored and independently checked.

## Driver recovery gate

After a host reboot, `nvidia-smi` PASSed with driver 580.173.02 and reported CUDA 13.0. This clears the driver
failure itself. Formal OpenVLA capture remains deferred because a separate `unitree_rl` compute process was using
3381 MiB, with total GPU use 3990/12227 MiB and 64% utilization. This is a contention boundary, not another P0-HD
failure; no model load, environment step or checkpoint check occurred in the recovery gate. Structured record:
`results/p0_hd_gpu_recovery_gate_20260828.yaml`.

At 00:41:05 the prior compute workload had exited. A second preflight PASSed with 620/12227 MiB used, 11607 MiB
derived free, 12% utilization and graphics processes only. The base-init5 single-EGL v2 presentation gate is now
eligible to run once. This resource PASS is not model/checkpoint or environment evidence; see
`results/p0_hd_gpu_resource_gate_20260828_004105.yaml`.

## OpenVLA frozen-base init5 gate

The single-sim/single-EGL v2 gate PASSed. Frozen base init5 reached strict success and reward 1.0 in 83 steps with 83
predictions. Model load was 4.619 s; inference total/mean/median/P95 was 28.437/0.3426/0.3369/0.3434 s, or 2.919
predictions/s. Peak allocated/reserved VRAM was 5,061,286,912/5,196,742,656 bytes. Internal elapsed was 57.46 s;
host wall was 61.67 s, maximum RSS 5,805,596 KiB, zero swap and zero socket messages. The worker reported the frozen
revision `fa5ae1e...baf`, loaded all four shards and shut down cleanly with 83 predictions.

Both videos contain 83 H.264/yuv420p frames at 20 fps: 1920x1080 presentation and 1080x1080 clean master. Visual QA
PASSed genuine 640-source detail, orientation and overlay layout. Training loss, offline validation/token accuracy,
L1, gradients, optimizer state and checkpoint creation are not applicable. This presentation recapture does not
revise the Phase 3A 45/45 final. The next gate is frozen-base init40 under a fresh v2 config; QLoRA follows only after
that base half of the paired display succeeds.

## OpenVLA frozen-base init40 gate

The base half of the typical-speedup pair PASSed at exactly 221 steps/reward 1.0, matching the frozen final
trajectory outcome. There were 221 predictions; load was 5.509 s and inference total/mean/median/P95 was
74.359/0.3365/0.3366/0.3434 s (2.972 predictions/s). Peak allocated/reserved VRAM remained
5,061,286,912/5,196,742,656 bytes. Host wall was 132.18 s, maximum RSS 5,775,076 KiB and swap/network were zero.
Revision/shards/worker cleanup PASSed.

Both hashed H.264 videos have 221 frames at 20 fps and visual QA PASSed. No offline or training metric applies. A
fresh single-EGL QLoRA init40 v2 config changes only the frozen worker/policy label; seed, state, policy preprocessing,
rendering and control contract match base. It is a presentation recapture of an already-finalized held-out state and
cannot be used for adapter selection or change the 43/45 final.

## OpenVLA QLoRA init40 gate

The adapter half PASSed at the frozen 70 steps/reward 1.0. Relative to base's 221 steps, this state uses 151 fewer
steps (-68.33%; base/adapter step ratio 3.16x), while adapter inference itself is slower: mean/median/P95
0.4599/0.4528/0.4627 s versus base 0.3365/0.3366/0.3434 s. Adapter load/integrity PASSed with SHA-256
`6b4fce2e...735c`; peak allocated/reserved VRAM was 5,468,073,984/5,584,715,776 bytes. Host wall was 63.53 s,
max RSS 5,848,468 KiB and swap/network were zero.

The 70-frame HD videos and trajectory pass hashes, media probe and visual QA. This supports a typical-state control
path acceleration demonstration only. It does not establish faster model inference or environment reliability and
cannot offset the frozen 43/45 versus 45/45 final. The next presentation pair is init44: base success first, then the
already-known adapter timeout with no tuning or checkpoint change.

## OpenVLA frozen-base init44 gate

Base init44 PASSed at the frozen 68 steps/reward 1.0. Load was 4.257 s; inference total/mean/median/P95 was
23.425/0.3445/0.3390/0.3447 s (2.903 predictions/s). Peak allocated/reserved VRAM was unchanged at
5,061,286,912/5,196,742,656 bytes. Host wall was 53.58 s, maximum RSS 5,776,412 KiB and swap/network were zero;
revision, four shards and cleanup PASSed. Both 68-frame videos pass media/hash/visual QA.

The only remaining formal P0-HD capture is frozen QLoRA init44 at the existing 600-step failure cap. Its fresh v2
contract uses the identical state/seed/native-128 policy input/direct-640 display protocol and the frozen adapter
worker. This is authorized only as presentation recapture of known final evidence; no success is expected, and it
must not trigger tuning, checkpoint replacement or a final rerun.

## QLoRA init44 and P0-HD completion

The final run is `EXPECTED_EXCEPTION`: execution PASSed and the policy reproduced the preregistered failure at the
600-step cap with reward 0.0. Model load was 5.189 s; inference total/mean/median/P95 was
291.498/0.4858/0.4843/0.5036 s (2.058 predictions/s). Peak allocated/reserved VRAM was
5,468,073,984/5,584,715,776 bytes; host wall 407.17 s, maximum RSS 5,872,152 KiB and swap/network were zero. Frozen
base revision and adapter SHA matched, all four shards loaded, 600 predictions completed and cleanup PASSed. The
30-second/600-frame videos and trajectory pass hashes, media probe and visual QA.

P0-HD is COMPLETE. Offline composition used zero environment steps and zero model predictions to produce two
1920x1080 comparisons in `outputs/presentation/p0_hd_comparisons_v1_20260828`: init40 typical speedup and init44
tail failure. Canonical records are `results/p0_hd_comparisons_v1_manifest.yaml` and the P0-HD sections of
`results/p0_mvp_v2_manifest.yaml`. The adapter remains research-only; 43/45, environment non-regression FAIL and all
no-retuning/no-final-rerun boundaries remain immutable.
