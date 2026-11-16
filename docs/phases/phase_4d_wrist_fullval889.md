# Phase 4D: paired wrist fullval889

Date: 2026-08-28  
Status: `PAIRED_OFFLINE_COMPLETE_ENVIRONMENT_GATE_PENDING`

## Purpose and boundary

Resolve Phase 4C's mixed val10 signal using the entire frozen 889-frame wrist validation set. Frozen base runs
first; only after its result/integrity audit may the step-20 wrist adapter run on the exact same fixture. Metrics are
mean frame loss, action-token accuracy and decoded continuous-action L1, plus per-episode values, throughput and
resources. These are teacher-forced offline metrics, never success or reward.

The fixture contains every frame from unchanged validation episodes `[0,5,12,21,29,34,40,42,43,45]`, sourced from
`obs/eye_in_hand_rgb` with identical action labels. Both evaluators reuse already-frozen Phase 3B implementations via
config-only wrappers. No training, checkpoint selection or environment execution occurs. Final tasks remain
untouched, and even favorable fullval does not automatically authorize a rollout.

## Static result

All 11 static checks PASSed: 889 wrist-only items, shared fixture hash and validation split, canonical base revision,
adapter hash, fresh base/adapter targets, frozen evaluator implementation hashes, config-only wrappers and explicit
read-only flags. The static run initialized no CUDA, processed no validation frames and executed no environment/model
work. Canonical result: `results/phase4d_openvla_wrist_fullval889_static.yaml`.

Execution order is immutable: frozen base first, then result/hash/resource audit, then adapter. They must not be run
in one unreviewed batch because a base failure would invalidate the paired comparison.

## Frozen-base result

Base fullval PASSed all 889 frames/6,223 action tokens with finite rows. Mean frame loss was 16.6834, action-token
accuracy 13.0805% and decoded continuous-action L1 0.28796. Model load/validation took 5.68/201.19 s; throughput was
4.419 frames/s and median/P95 forward latency 0.2235/0.2293 s. Peak allocated/reserved VRAM was
5,167,181,312/5,335,154,688 bytes. Host wall was 209.32 s, max RSS 5,768,068 KiB, zero swap/network.

Revision, fixture and all output hashes match; no training/environment/checkpoint selection occurred. This is the
paired offline baseline, not policy success or reward. Canonical result:
`results/phase4d_openvla_wrist_base_fullval889_v1.yaml`.

## Step-20 adapter and paired conclusion

The adapter PASSed the same 889 frames/6,223 tokens with finite rows. Loss improved from 16.6834 to 9.0479
(-45.77%), token accuracy from 13.0805% to 20.8099% (+7.73 percentage points; 814 to 1,295 correct tokens), and
continuous L1 from 0.28796 to 0.19459 (-32.43%). Loss improved on 884/889 frames and L1 on 756/889; all ten
validation episodes improved on all three aggregate metrics. This resolves the val10 ambiguity as a strong full-set
offline adaptation signal.

The adapter cost is measurable: median/P95 forward latency increased 15.69%/15.43%, throughput fell 13.28%, and
allocated/reserved VRAM rose 5.82%/4.68%. Host wall was 240.92 s, max RSS 5,846,720 KiB, with zero swap/network.
All revision, adapter, fixture and output hashes passed; no training, environment execution or checkpoint selection
occurred during validation.

Decision: `OFFLINE_ADAPTATION_PASS_ENVIRONMENT_RECOVERY_UNKNOWN`. Step-20 is eligible only as the fixed candidate
for a newly preregistered development rollout. Offline loss/token accuracy/L1 are not success or reward; final tasks
remain untouched. Canonical adapter and comparison records are
`results/phase4d_openvla_wrist_qlora_step20_fullval889_v1.yaml` and
`results/phase4d_openvla_wrist_step20_vs_base_fullval889_paired.yaml`.
