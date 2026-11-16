# Phase 4C: development-only wrist-view adaptation

Date: 2026-08-28  
Status: `DATA_AND_TRAINING_STATIC_PASS_HOST_PENDING`

## Hypothesis

Direct wrist-view substitution failed 0/15 in Phase 4B because the frozen policy was adapted to a fixed third-person
visual distribution. A task-specific rank-8 QLoRA trained on native `eye_in_hand_rgb` demonstration frames may
recover closed-loop wrist control for the development task `turn_on_the_stove`.

The first gate reuses the existing hashed 50-demo task dataset and frozen 40/10 episode split. Only the image source
changes from `agentview_rgb` to `eye_in_hand_rgb`; language, 7D action transform, NF4 base revision and QLoRA smoke
hyperparameters remain fixed. The initial candidate is exactly 20 updates plus a fixed 10-frame validation smoke.

This is single-task development work. Offline loss/token accuracy improvement is not success or reward, the smoke
cannot select a checkpoint, and final tasks `[1,2,4,5,9]` remain untouched. Environment evaluation requires a later
separately frozen gate and may not delete the Phase 4B 0/3 task-7 wrist baseline.

## Completed static evidence

The data contract PASSed all nine checks with zero CUDA/environment/model work. The frozen split contains 3,571
training and 889 validation wrist frames, all 128x128 uint8 RGB with nonzero channel variance and finite 7D actions.
Mean episode-level agent/wrist pixel MAE is 76.02. A deterministic 20-train/10-validation fixture was exported from
the native wrist key; all 30 items and its manifest are hashed.

The first training static check failed only because its checker contained an incorrect expected hash for the reused
Phase 3B trainer; it performed zero CUDA/training and is preserved. The corrected v2 static gate binds the actual
trainer SHA and PASSed all 12 checks, including canonical base revision, wrist-only fixture, split, rank-8/optimizer
contract, fresh outputs and wrapper-only config override. Expected peak reserved VRAM from the matched Phase 3B
smoke is 7,128,219,648 bytes.

Canonical records:

- `results/phase4c_libero_goal_wrist_data_contract_v1.yaml`
- `outputs/reference/phase4c_wrist_qlora_train20_val10_fixture_v1_seed20260828/manifest.json`
- `results/phase4c_openvla_wrist_qlora_train_smoke_v2_static.yaml`

The next host gate is formal training but remains only a 20-update development smoke. Even if offline metrics improve,
no environment-success or checkpoint-superiority claim is allowed.

## Training smoke result

Execution PASSed 20/20 updates. All losses, gradients and 879 adapter tensors were finite; optimizer/RNG state is at
step 20 with 852 optimizer entries. The last-ten train-loss mean (11.585) was 12.97% below the first-ten mean
(13.311), but gradients before clipping remained large (mean 438.8, range 361.6--533.1).

The fixed val10 signal was mixed: mean sample loss improved from 13.704 to 9.279 (-32.29%), while action-token
accuracy fell from 16/70 to 13/70 (-4.29 percentage points). Therefore the decision is
`PASS_ENGINEERING_MIXED_OFFLINE_SIGNAL_NO_CHECKPOINT_SELECTION`. No success, reward or environment inference is
available. The next gate must be paired full wrist validation over all 889 frozen validation frames before any
checkpoint decision or rollout protocol.

Model load/adapter preparation/train+validation took 5.69/0.35/21.34 s; host wall was 31.96 s. Peak allocated/
reserved VRAM was 6,866,078,208/7,065,305,088 bytes; host max RSS 5,776,904 KiB, zero swap/network. Adapter,
optimizer/RNG and all five checkpoint files passed hash/CPU integrity checks. Canonical structured result:
`results/phase4c_openvla_wrist_qlora_train_smoke_v1.yaml`.
