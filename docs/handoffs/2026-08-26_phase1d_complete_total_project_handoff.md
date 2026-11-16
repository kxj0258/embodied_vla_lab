# Embodied VLA Lab total handoff — Phase 1D complete

Date: 2026-08-26
Project root: `/home/kxj/embodied_ws/embodied_vla_lab`
Environment: `evla-core`
Branch: `main`
Latest local commit: `f0c0368fd07b51928bb1b96be2c7e0fc45818182`
Remote push status: NOT PUSHED

This is the latest total handoff. It supersedes earlier conversational handoffs for continuation purposes while
preserving them as historical evidence. The next conversation must read this file from beginning to end before
taking any project action.

## 1. Mandatory collaboration and authorization contract

- Do not use multi-agent.
- Use the `evla-core` environment.
- The user executes formal training and longer evaluations in the host terminal; provide one minimal command and
  wait for pasted output.
- Similar local read-only inspections are pre-authorized and may be executed directly.
- Installation, dataset/model/checkpoint download, any new training, GUI, W&B, Hub/upload, commit, and push each
  require separate explicit user authorization.
- Environment construction, GPU forward, and new evaluation should also be treated as new execution gates and
  explicitly authorized when they cross the current frozen phase boundary.
- Do not overwrite old logs, outputs, checkpoints, structured results, or manifests. Every new experiment uses a
  new ID, raw log, output path, structured result, and manifest entry.
- Existing Phase 1B/1C/1D final results are frozen. Do not reinterpret development results as final results.
- `env_backup_evla-core.yml` is a pre-existing unrelated untracked file. Preserve it and exclude it from project
  commits unless the user explicitly changes its status.
- After every pasted host result:
  1. classify it first as PASS, FAIL, or EXPECTED_EXCEPTION;
  2. analyze loss, offline eval, success, reward, gradients, throughput, VRAM, host memory, elapsed time, and the
     statistical conclusion boundary, explicitly marking non-applicable or unmeasured fields;
  3. update the structured result, checkpoint manifest, README, phase document, handoff, and reproducibility assets;
  4. end with exactly one next minimal safe host command, or stop at an explicit authorization boundary.

## 2. First checks for the next conversation

Run these read-only checks before proposing work:

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab
conda activate evla-core
git status --short
git rev-parse HEAD
git log -6 --oneline
git submodule status
sed -n '1,260p' README.md
sed -n '1,280p' results/phase1d_pusht_reference_final_100seed.yaml
sed -n '1,260p' results/phase1d_pusht_reference_checkpoint_manifest.yaml
scripts/runs/pusht.sh --list
```

## 14. Phase 2B 100k completion update (latest; 2026-08-27)

The formal experiment `phase2b_act_libero_turn_on_stove_deterministic_100k_seed20260826` completed all 100,000
optimizer steps and is classified PASS. The raw log ends with `classification: PASS`, process wall 3,354.04 s and
maximum RSS 2,558,808 KiB. Peak allocated/reserved VRAM was 709,839,360/834,666,496 bytes and overall optimizer
throughput was about 29.86 steps/s using process elapsed time.

The sparse logged train curve contains 1,001 points: loss 0.917988 at step 1, minimum 0.048785 at step 81,000,
final 0.070784, and last-ten mean 0.065300. Pre-clip gradient norm peaked at 89.894 on step 1 and had a last-ten mean
of 3.356. The clip boundary was 10; 65 sparse logged points exceeded it, but this is not the exact number of clipped
optimizer steps because only step 1 and every 100th step were logged.

All 101 full validations covered all 889 fixed validation frames. Offline loss was 1.016288 at step 0, reached the
run minimum 0.426634 at step 1,000, then remained broadly on a 0.43--0.47 plateau and ended at 0.440030. The complete
curve is frozen in `results/phase2b_act_libero_100k.yaml`. It is offline action loss only, not LIBERO success or reward,
and does not establish that the minimum-loss checkpoint is the best closed-loop policy.

Checkpoint integrity PASSed. There are 22 exact-resume periodic checkpoints from step 4,460 through 98,120 plus
one archival non-boundary step-100,000 checkpoint. All 23 are 410,470,545 bytes, CPU-loadable, contain all required
state fields, and have hashes in `results/phase2b_act_libero_checkpoint_manifest.yaml`; no `.pt.incomplete` remains.
The final checkpoint correctly stores `exact_resume_boundary: false`.

No trained-policy environment rollout was run. Success, reward, rollout FPS, failure categories, closed-loop
checkpoint selection and multi-seed statistics remain unknown. One training seed and one fixed 10-episode offline
validation split support only the finite-optimization/offline-generalization claim. The next work is a separately
reviewed, new-ID independent LIBERO checkpoint rollout entrypoint. Environment construction and evaluation remain at
an explicit authorization boundary; do not run them automatically. No commit or push was performed.

## 15. Phase 2B first trained-policy rollout update (latest; 2026-08-27)

`phase2b_act_libero_ckpt100000_dev5_seed20260827` is PASS. The valid archival step-100k checkpoint achieved strict
success on all five fixed development init states 0--4, each in 78--83 environment steps, with sum/max reward 1.0
for every episode. This validates the complete adapter and closed-loop control path; offline validation loss is not
being substituted for this environment evidence.

The run used 403 environment steps in 17.456 s (23.09 steps/s; 3.491 s/episode), 82 policy forward calls, and 0.989 s
total timed action selection including CUDA cold start. Warm episodes imply about 7.81 ms per forward. Peak
allocated/reserved VRAM was 162,552,832/216,006,656 bytes; process wall was 24.06 s and maximum RSS was 4,020,488
KiB. No training loss, offline validation, gradient or optimizer metric was remeasured.

The robosuite private-macro warning is non-semantic. With `HF_HUB_OFFLINE=1`, the asset helper printed its generic
download message and then confirmed the existing local cache; no new asset download was observed. The canonical
result is `results/phase2b_act_libero_ckpt100000_dev5.yaml` and all config/script/log/eval-info hashes are recorded.

Do not call this a final 100% success rate: N=5 yields a wide Wilson 95% interval (about 56.6%--100%), these states
are development evidence, and there is no checkpoint comparison. The next minimal paired gate is the same five
states on `model_004460.pt` under the new ID `phase2b_act_libero_ckpt004460_dev5_seed20260827`. If checkpoint
selection is then frozen, use a separate held-out final evaluation (Section 17 corrects its size to 45). No commit or push was performed.

## 16. Phase 2B early-checkpoint paired rollout update (latest; 2026-08-27)

`phase2b_act_libero_ckpt004460_dev5_seed20260827` is PASS for evaluation execution and artifact production. Its
policy achieved 4/5 strict successes on the same development init states used for step 100,000. States 0--3
succeeded in 74/72/115/163 steps; state 4 timed out at the 600-step horizon with zero reward. Mean sum/max reward
was 0.8. Gate classification and policy success rate are separate fields, so one failed episode does not make the
successfully completed evaluation a FAIL.

The paired comparison currently favors step 100,000: 5/5 versus 4/5, with the sole discordant state succeeding at
step 100,000 in 78 steps. All-episode mean steps were 80.6 versus 204.8. However, one discordant pair is insufficient
for a significance claim. This rollout did not remeasure loss, validation, gradients or optimizer throughput.

The early run used 1,024 environment steps, 206 policy forwards and 19.846 s pure rollout; peak allocated/reserved
VRAM was 162,552,832/216,006,656 bytes, process wall 26.47 s and maximum RSS 3,921,696 KiB. Canonical result:
`results/phase2b_act_libero_ckpt004460_dev5.yaml`.

To bound selection bias, do not evaluate all 23 checkpoints. The third and final development candidate is frozen as
the midpoint `model_049060.pt`, using the identical five states and new ID
`phase2b_act_libero_ckpt049060_dev5_seed20260827`. After it, select lexicographically by success, then reward, then
fewer completion steps, and freeze the selected checkpoint before a held-out final (Section 17 corrects its size to 45). No commit or push.

## 17. Phase 2B checkpoint selection frozen (latest; 2026-08-27)

The third bounded candidate `phase2b_act_libero_ckpt049060_dev5_seed20260827` is PASS and achieved 5/5 strict
success, mean sum/max reward 1.0 and 407 total environment steps. The three development candidates are therefore:
step 4,460 = 4/5 and reward 0.8; step 49,060 = 5/5, reward 1.0, 407 steps; step 100,000 = 5/5, reward 1.0, 403 steps.

The preregistered lexicographic rule (success, then reward, then fewer total steps) freezes `model_100000.pt`.
Its four-step advantage over step 49,060 is only a tie-break on five development states, not statistical
superiority. No more checkpoints may be evaluated for selection, and final evidence cannot revise this choice.
Canonical selection: `results/phase2b_act_libero_checkpoint_selection_dev5.yaml`.

Correct a protocol impossibility in earlier text: only 50 init states exist and indices 0--4 have already been used
for development, so there cannot be a never-used 50-state final. The preregistered uncontaminated final is all 45
remaining states, indices 5--49, under
`phase2b_act_libero_ckpt100000_final_heldout45_seed20260901`. Evaluator seeds are 20260901--20260945. The evaluator
now supports a nonzero sequential init-state start and preserves strict no-overwrite behavior. No final run, commit
or push was performed in this update.

## 18. Phase 2B held-out final complete (latest; 2026-08-27)

`phase2b_act_libero_ckpt100000_final_heldout45_seed20260901` is PASS. The checkpoint was frozen before final
execution and remained `model_100000.pt`. It achieved 45/45 strict BDDL successes on every init state excluded from
development, indices 5--49; every sum/max reward was 1.0. Selection must not be revised.

The 45 episodes used 3,584 environment steps. Mean/median completion steps were 79.64/79, range 74--106 and P95 87.
Pure rollout took 81.612 s (1.814 s/episode; 43.92 env steps/s). There were 733 policy forwards; mean timed selection
was 8.22 ms including cold start and 7.77 ms warm. Peak allocated/reserved VRAM was
162,552,832/216,006,656 bytes, process wall 87.83 s and maximum RSS 4,178,080 KiB.

This exactly enumerates the complete frozen 45-state held-out finite set. If generalized as Bernoulli samples, the
45/45 Wilson 95% interval is approximately 92.13%--100%. Development states 0--4 were also 5/5 for the selected
checkpoint, so full pinned-state coverage is descriptively 50/50, but only 45/45 is uncontaminated final evidence.
The result remains single-task, single-training-seed, fixed-runtime and unperturbed-vision evidence; it does not
establish broader LIBERO task/language/viewpoint/simulator generalization.

Canonical result: `results/phase2b_act_libero_ckpt100000_final_heldout45.yaml`. Raw log SHA-256 is
`56e8791bc158dd8fcd5fc098b8684b97302c662e1e7d3a135aa1370f9db01adb`; eval JSON SHA-256 is
`7308dd3d9ee4bed7a572a956ef8f86b314f88ed2caf2f8d1b07512e5fd130901`. Phase 2B is complete. No commit or push
was performed. The next roadmap phase is Phase 3A OpenVLA feasibility, which requires a separate model/download/GPU
authorization and must not begin automatically.

## 19. Phase 3A OpenVLA static contract started (latest; 2026-08-27)

The user authorized autonomous short read-only analysis and project-asset development, while retaining host execution
for environment installation, large downloads, long GPU inference/evaluation and training. The Phase 3A source-only
gate is PASS. Target checkpoint: official `openvla/openvla-7b-finetuned-libero-goal` revision
`fa5ae1e7509348889295bba8e08621d8b55e9baf`; pinned source remains
`c8f03f48af692657d3060c19588038c7220e9af9`.

The checkpoint is approximately 8B BF16 parameters / 15.1 GB logical files and was officially LoRA-finetuned for
60k steps on LIBERO-Goal. The frozen inference path is single agent-view 224x224 with center 90% crop, language
prompt, seven generated action tokens, `libero_goal` q01/q99 unnormalization and LIBERO gripper normalization plus
sign inversion. It is not a pure comparison with dual-camera/proprioceptive chunked ACT.

Do not mutate `evla-core`. Create isolated `evla-openvla` with Python 3.10, cu128 PyTorch, Transformers 4.40.1,
tokenizers 0.19.1, timm 0.9.10 and bitsandbytes 0.47.0. Current official bitsandbytes CUDA 12.8 Linux wheels include
`sm_120`; first use NF4 4-bit BF16 compute without FlashAttention, TensorFlow/RLDS or W&B. No target environment,
model cache, package installation, model download, GPU initialization or inference occurred in the static gate.

Canonical result: `results/phase3a_openvla_static_contract.yaml`. The next separately executed host gate is
`scripts/setup_phase3a_openvla_env.sh`, which refuses to overwrite an existing environment and ends with a tiny
CUDA BF16/NF4 kernel smoke plus package freeze. No model download is part of that command. No commit or push.

## 12. Phase 2B 100k launch handoff (latest; supersedes the next-step text above)

This file remains the single cumulative handoff for all project conversations. Sections 1-10 preserve the frozen
Phase 1B/1C/1D history and boundaries; Section 10A records the complete Phase 2A/2B continuation. This section is
the authoritative latest state as of 2026-08-26 immediately before the formal 100k host run.

### Repository state at handoff

- Git HEAD: `f0c0368fd07b51928bb1b96be2c7e0fc45818182`.
- submodule pins: LeRobot `8b256a6c0d4769c3cc3e7e98f04940126398a391`, LIBERO
  `8f1084e3132a39270c3a13ebe37270a43ece2a01`, OpenVLA `c8f03f48af692657d3060c19588038c7220e9af9`.
- `README.md` and `docs/roadmap.md` are modified. Phase 2 configs, scripts, results, phase document, checkpoint
  manifest, and this handoff are untracked. `env_backup_evla-core.yml` is an unrelated pre-existing untracked file.
- No commit or push was performed in this conversation.

### Frozen Phase 2 evidence

- Phase 2A task/data contract: `LIBERO_GOAL/turn_on_the_stove`, official 50-demo/4,460-frame HDF5, frozen 40/10
  episode split with 3,571/889 frames, 9D state, 7D action, dual 128x128 RGB, ACT chunk 20/execute 5.
- Demo-0 task-semantic replay PASSed, but archived next-state determinism remains FAIL (`0.507810 > 0.01`). This
  accepted dual conclusion must never be rewritten as an exact replay PASS.
- Two-step GPU resource gate PASSed; the 200-step single-batch threshold FAILed; unchanged 400-step threshold PASSed.
- Full-split smoke v1 FAILed only at checkpoint config serialization after 20 finite optimizer steps. The fix is
  `dataclasses.asdict`. Managed-environment v2 is EXPECTED_EXCEPTION because CUDA was inaccessible before training.
- Host v3 PASSed training, checkpoint serialization, and CPU reload. Validation v4 PASSed a complete 889-frame
  offline validation with step-20 loss `0.7429550438281254`. Its checkpoint is 410,450,149 bytes, SHA-256
  `1e5be537a28f126980aeda82dd84144f5f2af3f59b764d19cd0e833cd90b5d82`.
- Phase 2 has no trained-policy environment rollout success rate or reward result yet. Offline action loss must not
  be described as task success. There is only one training seed and no Phase 2 statistical comparison yet.

### Formal 100k experiment contract

- Experiment ID: `phase2b_act_libero_turn_on_stove_deterministic_100k_seed20260826`.
- Config: `configs/train/act_libero_phase2b_single_task_100k.yaml`.
- Entrypoint: `scripts/train_phase2b_act_libero.py`.
- Planned manifest: `results/phase2b_act_libero_100k_run_manifest.yaml`.
- Output: `outputs/train/phase2b_act_libero_turn_on_stove_deterministic_100k_seed20260826`.
- Raw log: `results/raw/phase2b_act_libero_turn_on_stove_deterministic_100k_seed20260826_host_20260826.log`.
- 100,000 optimizer steps, batch 8, LR `1e-5`, weight decay `1e-4`, BF16, clip norm 10, strict offline,
  W&B disabled, Hub disabled. This is approximately 224.2 passes over the 3,571-frame train split.
- Full 889-frame offline validation runs at step 0 and every 1,000 steps.
- Periodic checkpoints are every 4,460 steps (ten complete 446-step epochs). They contain model, optimizer, ACT
  config, frozen statistics, Python/NumPy/PyTorch/CUDA RNG states, and DataLoader generator state.
- Checkpoints are written to `.pt.incomplete` and atomically renamed. Only periodic 4,460-step boundaries are
  accepted for exact continuation. The final step-100,000 checkpoint is archival/non-boundary and is not accepted
  as an exact-resume source.
- Estimated time is 50-100 minutes. Approximately 23 checkpoints require about 9.5 GiB; the prelaunch filesystem
  had 90,571,993,088 bytes free.
- Periodic evaluation inside training means offline validation loss only. LIBERO success/reward rollouts are kept
  out of the training process and must later run as separate new-ID checkpoint evaluations.

The initial 100k command had been issued to the user but no output from it existed when this handoff was written.
Therefore the next conversation must first determine whether the process PASSed, FAILed, was interrupted, or is
still running. Never infer that training started or completed merely because the output directory exists.

If interrupted, inspect the log and output read-only, ignore any `.pt.incomplete` file, verify size/hash/loadability
of the latest `model_*.pt`, and resume only from the latest checkpoint whose step is divisible by 4,460 and whose
stored `exact_resume_boundary` is true. Use a new resume log ID and set `EVLA_RESUME_CHECKPOINT`; do not reuse the
initial no-overwrite command. If complete, freeze all checkpoint hashes and validation/train/grad/resource/time
curves before selecting checkpoints for independent rollout evaluation.

### Required handling of the user's next terminal output

1. Lead with exactly one classification: PASS, FAIL, or EXPECTED_EXCEPTION. An intentional Ctrl-C is not PASS;
   describe it as an interrupted run after checking whether a valid recovery boundary exists.
2. Analyze train loss, complete offline validation curve, success, reward, pre-clip gradients and clipping boundary,
   optimizer throughput, allocated/reserved VRAM, maximum RSS, process/training time, checkpoint integrity, and the
   limits of any statistical conclusion. Do not invent omitted metrics.
3. Update the structured 100k result, `results/phase2b_act_libero_checkpoint_manifest.yaml`, README, roadmap, Phase
   document, this cumulative handoff, raw-log/checkpoint hashes, and other reproducibility assets. Frozen Phase 1
   and earlier Phase 2 result files must not be overwritten by a new experiment.
4. If interrupted, provide exactly one minimal host resume command using the verified checkpoint. If complete,
   provide exactly one minimal independent LIBERO rollout-evaluation command or stop at an authorization boundary.

## 13. Suggested opening message for the next conversation (current)

Copy the following message and then append the final 100k terminal output beneath it:

```text
请继续 Embodied VLA Lab 项目。不要使用 multi-agent。

项目根目录：
/home/kxj/embodied_ws/embodied_vla_lab

请先从头到尾完整读取这份累计总交接文档，尤其是最新的第12节和第13节：
/home/kxj/embodied_ws/embodied_vla_lab/docs/handoffs/2026-08-26_phase1d_complete_total_project_handoff.md

读取后，先执行其中要求的只读核对，不要自动启动训练、评测、下载、安装、GUI、W&B、Hub/upload、
commit、push 或删除。Phase 1B/1C/1D final 和已经记录的 Phase 2 gate 结果均已冻结；任何新运行必须使用
新 ID、日志、输出目录、结构化结果和 manifest 条目。

我接下来会粘贴 Phase 2B ACT-on-LIBERO 100k 正式训练的完整宿主终端输出。收到后请：
1. 首先判定 PASS、FAIL 或 EXPECTED_EXCEPTION；如果是人为中断，检查是否存在有效精确恢复边界；
2. 分析 train loss、完整 offline validation 曲线、success、reward、grad、吞吐、显存、主存、时间、
   checkpoint 完整性和统计结论边界；
3. 更新结构化结果、checkpoint manifest、README、roadmap、phase 文档、累计 handoff 和可复现资产；
4. 若中断，只给基于最新已验证 4,460-step 边界 checkpoint 的一条恢复命令；若完成，只给下一条最小、
   安全、独立的 LIBERO checkpoint rollout 评测命令，或明确停在授权边界。

短时本地读取、哈希、CPU checkpoint 加载和项目文件写入可直接执行。正式训练和较长环境评测由你给
命令、我在宿主执行并反馈。不要把 offline validation loss 当作环境 success 或 reward。
```

Expected Git facts at handoff creation:

- HEAD: `f0c0368fd07b51928bb1b96be2c7e0fc45818182`
- latest commit: `f0c0368 experiment: reproduce official PushT diffusion checkpoint`
- submodules:
  - LeRobot: `8b256a6c0d4769c3cc3e7e98f04940126398a391`
  - LIBERO: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
  - OpenVLA: `c8f03f48af692657d3060c19588038c7220e9af9`
- only expected untracked item before this new handoff is `env_backup_evla-core.yml`; this handoff itself will also
  be untracked until a separately authorized commit.
- Phase 1D is committed locally but has not been pushed.

Some Phase 1D structured YAML files contain `commit_created: false` because they were frozen immediately before the
commit. That field is a run-record snapshot, not the current Git state. Current Git truth is the HEAD above.

## 3. Stable project foundation

Hardware and runtime:

- Ubuntu 22.04.5 LTS;
- NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12 GB VRAM, Blackwell compute capability 12.0 / `sm_120`;
- 32 GB RAM;
- Python 3.12.14;
- PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128;
- pinned editable LeRobot 0.6.2;
- gym-pusht 0.1.6;
- headless/EGL simulation; no GUI required.

Frozen PushT dataset/runtime contract:

- dataset: `lerobot/pusht` revision `7628202a2180972f291ba1bc6723834921e72c19`;
- 206 episodes;
- 185 training episodes / 23,043 frames;
- 21 validation episodes / 2,607 frames;
- PushT-v0, RGB 96x96, 2D agent state, 2D absolute target action;
- 10 Hz, maximum 300 steps;
- runtime config: `configs/env/pusht_96.yaml`.

Phase 0 and Phase 1A completed environment, dataset, split, replay, offline contract, GPU, and reproducibility gates.
MuJoCo EGL and LIBERO offline reset/step plus dual-camera RGB rendering are already verified, but no LIBERO training
or task-level benchmark is frozen yet.

## 4. Frozen local training baselines

### 4.1 ACT

The project trained and diagnosed VAE ACT and deterministic ACT variants. Important facts:

- VAE ACT formal training reached 100k steps, but checkpoint selection showed closed-loop quality was non-monotonic;
- the frozen VAE comparison checkpoint is 25k, not automatically the last checkpoint;
- deterministic ACT 30k is the strongest frozen ACT sparse-success baseline;
- offline loss improvements did not guarantee closed-loop success;
- late pose correction, success retention, and terminal refinement were recurrent failure modes.

Final independent results on the historical 100-seed set `20260830..20260929`:

| Model | Success | Mean sum reward | Mean max reward | Eval s/episode |
|---|---:|---:|---:|---:|
| deterministic ACT 30k | 6/100 | 72.3539 | 0.51557 | 1.168 |
| VAE ACT 25k | 2/100 | 67.3235 | 0.41226 | 1.394 |

The deterministic-versus-VAE sparse-success difference was not statistically stable on paired seeds, although
deterministic ACT improved dense reward more consistently.

### 4.2 Local Diffusion Policy (Phase 1C)

The frozen local EMA 30k checkpoint was a lightweight, ACT-compatible model-family comparison, not an official
Diffusion Policy recipe reproduction. It used horizon 48, execute 10, no crop, no GroupNorm, batch 8, 30k steps,
and a 100-step DDPM.

Final historical 100-seed result:

- success: 4/100;
- mean sum reward: 86.8430;
- mean max reward: 0.65133;
- evaluation: 45.55 seconds/episode;
- Diffusion improved dense reward/proximity over both ACT baselines but did not improve sparse success over
  deterministic ACT;
- sparse success comparisons between local ACT/Diffusion were not statistically stable.

Do not claim that the local Diffusion model proves Diffusion Policy is inferior. It proves only that this lightweight
30k recipe did not deliver reliable sparse success.

## 5. Phase 1D official/reference reproduction

Reference identity:

- Hub repository: `lerobot/diffusion_pusht`;
- frozen revision: `84a7c23178445c6bbf7e1a884ff497017910f653`;
- source `model.safetensors`: 1,050,862,408 bytes;
- source model SHA-256: `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`;
- official recipe: batch 64, 200k steps, seed 100000, horizon 16, execute 8, random 84x84 training crop / center
  evaluation crop, GroupNorm, ImageNet visual statistics, state/action MIN_MAX, DDPM 100, epsilon prediction.

Compatibility findings and gates:

1. The official bundle lacks current `policy_preprocessor.json` and `policy_postprocessor.json`; direct pinned
   LeRobot evaluation was blocked before migration.
2. All 12 published config contracts matched.
3. The archive has 221 tensors: eight legacy normalization buffers plus 213 core model tensors.
4. Archived and pinned SpatialSoftmax `pos_grid` are both `[9,2]`; the reported `[144,2]` risk was not reproduced.
5. Normalization values are finite and complete: ImageNet RGB mean/std, state min/max, and identical action
   target/output min/max `[12,25]` / `[511,511]`.
6. CPU schema gate matched all 213 core tensors with zero missing/unexpected/shape mismatches.
7. Strict GPU load matched all 213 tensors, placed 262,709,026 parameters on CUDA, and used about 1008.3 MiB peak
   allocated memory for that isolated load gate.
8. The first upstream migration attempt failed at config serialization because raw JSON lists for `crop_shape`,
   `down_dims`, and `optimizer_betas` were not restored to tuple-typed fields. Its partial directory has a zero-byte
   config and no model and must never be evaluated.
9. The project-local controlled v2 migration used normal config loading and strict core loading. Independent
   verification proved all 213 migrated tensors bit-exact to source and reloaded policy/preprocessor/postprocessor
   offline.
10. A CUDA single-observation/action/environment-step smoke passed: finite in-bounds action, one accepted step,
    1.613 s cold action selection, about 1104.1 MiB peak allocated VRAM.

Valid migrated bundle:

`outputs/reference/phase1d_pusht_diffusion_official_84a7c231/migrated_lerobot_0_6_2_v2`

Invalid preserved partial bundle:

`outputs/reference/phase1d_pusht_diffusion_official_84a7c231/migrated_lerobot_0_6_2`

Never use the invalid directory for evaluation.

## 6. Latest formal result and interpretation

The latest result is an evaluation of the official pretrained reference checkpoint. It is not new training, so
training loss, offline validation loss, gradient norm/clipping, optimizer throughput, and training VRAM are not
applicable.

Development gate:

- 5 episodes from evaluator seed 20261010;
- 4/5 successes;
- mean sum reward 99.9626;
- mean max reward 0.999481;
- 35.33 seconds/episode;
- Wilson 95% interval 37.6%--96.4%; useful as a strong development signal, not a final estimate.

Preregistered final gate:

- never-used fixed range: `20261101..20261200`;
- 100 episodes;
- 63/100 strict successes;
- Wilson 95% interval: 53.22%--71.82%;
- mean sum reward: 106.62654465239943;
- mean max reward: 0.9470129047201568;
- sum reward range: 0.0--233.6608;
- max reward range: 0.0--1.0;
- pure evaluation time: 3020.15 s;
- 30.2015 seconds/episode;
- process wall time: 50:26.68;
- maximum RSS: 3,387,556 KiB;
- swaps and socket traffic: zero;
- peak VRAM was not recorded by the evaluator and must not be inferred from the single-step gate.

Compared with the three frozen 100-episode local baselines, the reference success rate is higher by 57 points over
deterministic ACT, 61 points over VAE ACT, and 59 points over local Diffusion. One-sided Fisher exact p-values are
approximately `7.8e-19`, `4.8e-23`, and `9.4e-21`. These tests assume independent Bernoulli episodes and compare
different fixed seed sets. The conclusion that the reference is decisively above the single-digit baselines is robust.

Ten failed reference episodes still reached max reward at least 0.99. All 63 strict successes reached max reward
exactly 1.0. Dense reward is diagnostically useful, but near-one reward must not be substituted for the evaluator's
strict success field.

The reference evaluated at 30.20 s/episode versus 45.55 s/episode for local Diffusion, an observed 33.7% reduction.
This is not a pure latency ablation: horizon/action window, checkpoint architecture, recipe, trajectories, and fixed
seed sets differ.

Strongest justified project conclusion:

- the current PushT evaluator and success-accounting stack can reproduce a strong public policy;
- the stack is not inherently capped at single-digit success;
- the official recipe/checkpoint is decisively stronger than the frozen local baselines;
- Phase 1C's low success should be interpreted as a local recipe/training/checkpoint problem, not proof of a broken
  evaluator or a general failure of Diffusion Policy;
- no single causal factor is isolated because training budget, batch, crop, GroupNorm, horizon/action window,
  normalization, and checkpoint provenance changed together.

## 7. Canonical evidence and reproducibility assets

Read these first when details are needed:

- `README.md`
- `docs/phases/phase_1d_pusht_reproduction.md`
- `results/phase1d_pusht_reference_checkpoint_metadata.yaml`
- `results/phase1d_pusht_reference_checkpoint_manifest.yaml`
- `results/phase1d_pusht_reference_dev_5ep.yaml`
- `results/phase1d_pusht_reference_final_100seed.yaml`
- `scripts/runs/pusht.sh`
- `experiments/pusht_reproduction/README.md`
- previous cumulative handoff:
  `docs/handoffs/2026-08-26_pusht_phase1c_complete_reproducibility_handoff.md`

Final raw evidence:

- log: `results/raw/phase1d_pusht_reference_final_seed20261101_100ep.log`
- log SHA-256: `66872f08b3154cb54cb239e868b76c87a3a8231d9977a6320104e2c5a1ce7fd1`
- eval JSON:
  `outputs/eval/phase1d_pusht_reference_final_seed20261101_100ep/eval_info.json`
- eval JSON SHA-256: `3b4fb1e93f815aee8fa39abf80265cf5caf3b8fdd52e939f84a12ebb5c2162d9`
- ten rendered videos are hashed in `results/phase1d_pusht_reference_final_100seed.yaml`.

The controlled entrypoint contains strict-offline, overwrite-protected profiles for all frozen ACT/Diffusion gates,
Phase 1D migration verification, single-step smoke, development evaluation, and final evaluation. Use
`EVLA_DRY_RUN=1 scripts/runs/pusht.sh <profile>` before any authorized rerun.

## 8. Git and workspace boundary

Phase 1D was committed locally as:

`f0c0368 experiment: reproduce official PushT diffusion checkpoint`

It has not been pushed. Push still requires separate explicit authorization. The new handoff created by the current
conversation is not included in `f0c0368` and requires another separately authorized commit if the user wants it
frozen in Git.

Do not add `env_backup_evla-core.yml` automatically. Downloaded source/bundle payloads, raw logs, and evaluation
outputs are intentionally outside the Git commit but are represented by structured hashes.

## 9. Next smallest safe and valuable work

Phase 1D is complete. The next project direction in the roadmap is a LIBERO single-task data/action contract gate,
not another PushT training run.

Recommended next work, in order:

1. Read-only inventory of the pinned LIBERO task registry, demonstration availability, action representation,
   observation/camera keys, success condition, and local cached assets.
2. Select exactly one short single-task benchmark and freeze its task identity, language instruction, demo revision,
   observation/action shapes, control frequency, episode horizon, reset/step/render contract, and seed policy.
3. Create inert Phase 2A documentation/config/preflight assets.
4. Request separate authorization before any dataset/demo download, environment rollout beyond read-only/local
   preflight, GPU forward, training, or evaluation.

Do not automatically begin ACT-on-LIBERO training. Do not start OpenVLA download/inference yet. A PushT
denoising-step latency ablation remains a possible later study, but it is lower value than establishing the first
language-conditioned LIBERO contract now that the PushT evaluator question is resolved.

## 10. Operations that still require explicit authorization

- push of commit `f0c0368`;
- commit/push of this new total handoff;
- installation or environment mutation;
- any dataset, model, demonstration, or asset download;
- any new training;
- any new formal or long evaluation;
- GPU forward or environment execution beyond already frozen evidence;
- GUI/video playback;
- W&B;
- Hub/upload or any remote write;
- deletion of the invalid partial migration directory or any other artifact.

## 10A. Phase 2A continuation update

After the Phase 1D handoff was read and verified, a local read-only LIBERO inventory selected
`LIBERO_GOAL/turn_on_the_stove` as the smallest single-task contract candidate. Its BDDL goal is
`(And (Turnon flat_stove_1))`; the BDDL SHA-256 is `cbd2a051...decc13`. The pinned pruned init-state tensor is
`[50,79]` float64 with SHA-256 `ed121db3...b91340`. The source-derived candidate runtime uses Panda/OSC_POSE,
7D normalized delta-pose plus gripper actions, 20 Hz, dual 128x128 RGB, five settle steps, and a 600-policy-step
evaluation cap.

No LIBERO demonstration HDF5 is local. The expected relative path is
`libero_goal/turn_on_the_stove_demo.hdf5`; its revision, hash, episode count, frame count, action distribution, and
data observation shapes remain unknown. `/home/kxj/.cache/libero/assets` is present (about 409 MiB), while
`LIBERO_CONFIG_PATH` is unset and `~/.libero/config.yaml` is absent. Avoid importing the pinned top-level LIBERO
module without an isolated existing config because it can prompt and write user-home configuration.

New inert assets are `configs/env/libero_goal_turn_on_stove_phase2a.yaml`,
`scripts/check_phase2a_libero_static_contract.py`, `results/phase2a_libero_single_task_static_contract.yaml`,
`results/phase2a_libero_single_task_contract_manifest.yaml`, and
`docs/phases/phase_2a_libero_single_task_contract.md`. Similar project file writes are now pre-authorized, but
download, environment execution, GPU forward, training, evaluation, commit, and push retain their separate gates.

The user subsequently authorized official demonstration metadata lookup and download. The official dataset repo is
`yifengzhu-hf/LIBERO-datasets` at `f13aa24a3da8c43c7225569f28c562979fa0e35a`. The canonical single-task HDF5 is
447,509,922 bytes with SHA-256 `387fc107...aac73d5`; it contains 50 episodes / 4,460 frames, dual 128x128 RGB,
finite 7D actions within `[-1,1]`, and one recorded terminal reward/done per episode. A bad redirected-resume artifact
is preserved as `.invalid-resume-duplicate` and must never be used. Static and data contracts now PASS; isolated
runtime replay still requires its separate environment-execution authorization.

The authorized `demo_0` offline EGL replay executed all 80 recorded actions and reached first success at step 70,
ending successful with reward sum 11 and valid dual-camera observations. However, the preregistered archived
next-state maximum L2 threshold failed (`0.507810 > 0.01`), with both qpos and qvel drift beginning after action 1.
Classify the overall deterministic replay gate FAIL while separately recording interface/task-semantic replay PASS.
Do not relax the observed threshold. Canonical result:
`results/phase2a_libero_demo0_replay_preflight.yaml`. Before training, make an explicit compatibility decision about
whether exact legacy simulator-state replay is required. New environment executions still require authorization.

The user accepted the dual compatibility conclusion and authorized rapid progression. Phase 2A then froze a
seed-20260826 40/10 demonstration split (3,571/889 frames), train-only 9D state and 7D action statistics, dual-camera
ACT inputs, 20-step action chunks, and padding semantics. The dataset checker PASSed. A two-step ACT GPU resource
attempt failed before model construction because CUDA is not visible in the managed container; run the frozen gate
on the host next. This is not a model/data failure and provides no loss/grad/VRAM evidence.

The host GPU resource rerun PASSed with 34.19M parameters, loss 0.6174 to 0.4564 in two steps, finite but high
pre-clip gradients 112.81/86.73 (clip 10), 2.994 steps/s, and 706,334,208-byte peak allocated VRAM. Proceed to a
single-batch overfit gate before formal training. The scalar-conversion warning was non-semantic and the script now
uses detached tensors.

The 200-step single-batch overfit run is FAIL against its frozen threshold: loss 0.617429 to 0.175013 is a 71.65%
reduction, short of the required greater-than-75%. The script's premature PASS line is invalid because the process
exited 1; reporting order is fixed. Run a new 400-step ID with the same threshold before formal training.

The new 400-step run PASSed the unchanged gate: loss 0.617429 to 0.135600 (78.04%), minimum 0.122852, final
pre-clip grad 14.98, 49.22 steps/s, and 706,334,208-byte peak allocated VRAM. A strict-offline full-split 10k ACT
entrypoint is now prepared; run a new-ID short smoke before starting the formal output directory.

Full-split smoke v1 executed 20 steps with finite loss/grad and about 677 MiB peak allocated VRAM, then FAILed before
checkpoint creation because pinned `ACTConfig` lacks `to_dict()`. The failed output is preserved. The writer now uses
`dataclasses.asdict`; rerun a new v2 smoke before formal 10k training.

The attempted v2 run inside the Codex-managed execution context is an EXPECTED_EXCEPTION: CUDA peak-memory reset
failed before any batch, forward, backward, or optimizer step. It provides no evidence about the serialization fix.
Its log and resolved config are preserved; use a new v3 ID for the host CUDA rerun.

Host v3 PASSed: 20 optimizer steps, finite train loss/gradients, successful checkpoint serialization, and CPU
checkpoint reload. The checkpoint is 410,450,149 bytes with SHA-256 `1e5be537...b5d82`; peak allocated/reserved
VRAM was 709,839,360/834,666,496 bytes, process wall 6.89 seconds, and maximum RSS 2,454,016 KiB. It has no offline
validation or rollout evidence. The training entrypoint now implements deterministic full validation-set loss;
validate that new path under another new smoke ID before medium-scale training.

Validation smoke v4 PASSed on all 889 frozen validation frames: step-20 loss 0.742955, peak allocated/reserved VRAM
709,839,360/834,666,496 bytes, process wall 8.61 seconds, and maximum RSS 2,475,580 KiB. Its checkpoint is
byte-identical to v3, so evaluation was non-mutating. This is not rollout success/reward evidence. The entrypoint
now logs step-0 validation and writes environment overrides into resolved config. Next run a 1,000-step medium gate
with validation at steps 0/250/500/750/1000 and checkpoints at 500/1000 before authorizing formal 10k.

The user then explicitly selected a 100k formal run. The training entrypoint now persists model, optimizer, all RNG
states, and DataLoader generator state. Checkpoints every 4,460 steps are exact-resume epoch boundaries; offline
validation runs every 1,000 steps. The final 100k checkpoint is not an exact-resume boundary. Success/reward rollouts
must remain separate checkpoint evaluations rather than sharing the training process.

## 11. Suggested opening message for the next conversation

Copy the following message verbatim into a new conversation:

```text
请继续 Embodied VLA Lab 项目。

项目根目录：
/home/kxj/embodied_ws/embodied_vla_lab

请先从头到尾完整读取最新总交接文档：
/home/kxj/embodied_ws/embodied_vla_lab/docs/handoffs/2026-08-26_phase1d_complete_total_project_handoff.md

读取后，严格执行其中“First checks for the next conversation”的只读核对，并先向我报告：
1. 当前 Git HEAD、三个 submodule pins、工作区状态和未跟踪文件；
2. Phase 1B/1C 冻结训练基线，以及 Phase 1D 官方 reference 的 63/100 final 结果；
3. 最新结果对 loss、eval、success、reward、grad、吞吐、显存、时间和统计结论的边界；
4. Phase 1D 已解决的问题、仍不能归因的问题；
5. 下一步最小、安全、最有价值的 LIBERO 单任务只读合同工作；
6. 哪些操作仍需我的明确授权。

后续我会继续粘贴宿主终端输出。每次收到输出后必须：
- 首先判定 PASS、FAIL 或 EXPECTED_EXCEPTION；
- 分析 loss、offline eval、success、reward、grad、吞吐、显存、主存、时间和统计结论边界；
- 更新结构化结果、checkpoint manifest、README、phase 文档、handoff 和可复现资产；
- 最后只给下一条最小、安全的宿主命令，或明确停在授权边界。

正式训练和较长评测仍由你给出命令、我在宿主执行并反馈。类似本地只读检查可以自行执行，使用
evla-core。不要使用 multi-agent。不要因为读取交接文档而自动启动训练、评测、下载、安装、GUI、
W&B、Hub/upload、commit、push 或删除文件。安装、下载、新训练、GUI、W&B、Hub/upload、commit 和 push
都需要分别获得我的明确授权。Phase 1B/1C/1D final 结果已经冻结，任何新实验必须使用新 ID、日志、
输出目录、结构化结果和 manifest 条目。
```

## 20. Phase 3A environment and frozen model download complete (latest; 2026-08-27)

The isolated `evla-openvla` environment gate PASSed with Python 3.10.21, PyTorch 2.11.0+cu128, Transformers 4.40.1
and bitsandbytes 0.47.0. A finite NF4 `Linear4bit` kernel executed on the RTX 5070 Ti Laptop GPU (`sm_120`). This was
only a tiny quantization-kernel test, not OpenVLA inference.

The subsequent official checkpoint download is also PASS. Exactly 16 files totaling 15,085,050,797 logical bytes
were fetched at frozen revision `fa5ae1e7509348889295bba8e08621d8b55e9baf`; the four safetensors shards total
15,082,600,832 bytes. Every file hash is frozen in the model manifest. Config and statistics confirm a 224x224
OpenVLA policy, seven action dimensions, 256 bins, and `libero_goal` q01/q99 statistics based on 52,042 transitions
and 428 trajectories. Download elapsed/process wall were 4,027.64/4,027.80 s, maximum RSS was 487,816 KiB, and
56,123,138,048 filesystem bytes remained.

Canonical records are `results/phase3a_openvla_download_gate.yaml` and
`results/phase3a_openvla_checkpoint_manifest.yaml`. Raw log SHA-256 is
`d4a95a3dc5d97bdd8a45f33f62e0d4bc83fc2e4aec8af753ab5b3b26ea0480f4`; generated download manifest SHA-256 is
`cb0e189abfdf0888008f19fbf48d3c3215968acfe43098a9d2667457adb728fc`.

This establishes checkpoint identity/completeness only. The model was not constructed, CUDA was not initialized by
the download script, and there is no OpenVLA VRAM, action validity, latency, environment success/reward, or training
evidence. Next: prepare and host-run one new-ID NF4 4-bit load plus exactly one action prediction on a frozen local
LIBERO agent-view image. Do not begin rollout, QLoRA, training, upload, commit, push, or deletion automatically.

## 21. Phase 3A local inference-class imports complete (latest; 2026-08-27)

The first host import gate is FAIL even though installation of draccus 0.8.0, rich 13.9.4 and their small transitive
dependencies succeeded. Importing `prismatic.extern.hf` executed the upstream top-level package initializer, which
eagerly reached training-only RLDS code and raised `ModuleNotFoundError: dlimp`. Process wall was 10.52 s and maximum
RSS 804,916 KiB. No model was loaded and this is not checkpoint, GPU, inference, success/reward, or training evidence.
Do not install the full TensorFlow/RLDS/dlimp stack for this inference gate.

The inference scripts now register namespace paths only for the four required local HF classes, bypassing training
initializers without modifying the pinned OpenVLA submodule. The final new-ID v3 checker is PASS: draccus 0.8.0,
rich 13.9.4, all four classes imported, 2.21 s wall, 811,788 KiB maximum RSS, `cuda_initialized=false`, and no model
load. The intermediate v2 class import worked but reported a stale package-internal draccus version, so it is retained
only as invalid-version-reporting history; v3 reads installation metadata. Canonical results:
`results/phase3a_openvla_local_source_imports.yaml` (host v1 FAIL) and
`results/phase3a_openvla_local_source_imports_v3.yaml` (final PASS). Next host gate is exactly one NF4 4-bit load and
one action prediction on the frozen real LIBERO demo-0 agent-view frame; no environment construction or rollout.

## 22. Phase 3A first single-image load attempt failed before model load (latest; 2026-08-27)

`phase3a_openvla_4bit_single_image_demo0_t000_seed20260827` is FAIL at processor construction. The checkpoint's
`auto_map` points to `openvla/openvla-7b--processing_prismatic.*`; setting `trust_remote_code=true` therefore bypassed
the registered pinned local classes and attempted an offline lookup for `processing_prismatic.py`. The file was not
in that separate Hub cache, so the command exited 1 after 2.44 s with maximum RSS 834,704 KiB. No weight shard was
loaded, CUDA was not initialized, and there is no model-load, VRAM, latency, action, rollout, success/reward, or
training evidence. Raw log SHA-256: `f17ffb0aa57877deebbfb11e7cdbf961e62326c9b846ecc373324a8f72f74dd5`.

Canonical failure record: `results/phase3a_openvla_4bit_single_image.yaml`. The v2 entrypoint uses a new experiment
ID and `trust_remote_code=false` after registering exactly the four inference classes from pinned local source. This
forces local AutoClass mappings and preserves strict offline execution. Recheck processor construction without CUDA
before asking the host to retry the full model load; do not overwrite the failed v1 log or output identity.
That processor-only recheck is now PASS: the expected Prismatic processor/image processor and LlamaTokenizerFast
resolved entirely locally, with `cuda_initialized=false` and no model load. The new-ID v2 full host load may proceed.

## 23. Phase 3A v2 reached weight load but failed at quantized dispatch (latest; 2026-08-27)

The v2 command loaded all four checkpoint shards in approximately 10 s, then FAILed because Accelerate 1.10.1 called
`.to(device)` during single-device dispatch. Frozen Transformers 4.40.1 explicitly forbids `.to()` for bitsandbytes
4-bit models. This is a version-interface mismatch, not OOM: process wall was 13.91 s and maximum RSS 5,833,868 KiB.
The process exited, releasing its GPU allocations; no usable model was returned, inference did not run, and no action,
latency, success/reward, or rollout evidence exists. Raw log SHA-256 is
`5e2613d2c165d06a2880151727585498f3ec619255dd1398ac5d25efadd81394` and the canonical record is
`results/phase3a_openvla_4bit_single_image_v2.yaml`.

Keep all frozen versions unchanged. The new-ID v3 entrypoint installs no package and applies a scoped compatibility
shim: call the same Accelerate dispatcher with `force_hooks=true`, preventing its terminal `.to()` while retaining
the already loaded CUDA placement. Retry only the single frozen-image action gate; do not construct LIBERO or train.

## 24. Phase 3A v3 exposed exact one-token attention-mask bug (latest; 2026-08-27)

The force-hooks compatibility shim succeeded: v3 loaded all four shards, completed model dispatch and entered the
first generation forward. It then FAILed because attention logits had key length 276 while the causal mask had 275.
A processor-only reproduction established the exact arithmetic. The prompt has 19 input IDs and 19 mask entries and
ends in token 29901. Upstream `predict_action()` conditionally appends required token 29871 to `input_ids` but does
not extend `attention_mask`; after inserting 256 visual patch tokens, lengths are therefore 20+256=276 and
19+256=275. This is not OOM, quantization corruption, or a Blackwell kernel failure. Process wall was 10.35 s,
maximum RSS 5,862,552 KiB, and no action was produced. Raw log SHA-256 is
`b1b64df863502e38da7b2a1cfb70f49a7e4f2a65d749fc6d6e4736a4825d1e35`.

Canonical record: `results/phase3a_openvla_4bit_single_image_v3.yaml`. The new-ID v4 caller appends token 29871 and
one true mask entry atomically before calling `predict_action`; the upstream conditional append then becomes a no-op.
Retain the strict-offline local classes, NF4 settings and force-hooks dispatch shim. Do not change package versions,
weights, attention semantics, environment, or training until this single-action gate completes.

## 25. Phase 3A NF4 single-image action prediction PASS (latest; 2026-08-27)

`phase3a_openvla_4bit_single_image_v4_demo0_t000_seed20260827` is PASS. The exact frozen official revision loaded in
5.806 s and generated one finite 7D action from the hash-frozen real LIBERO demonstration frame in 1.024 s. The
prompt/token alignment fix was applied and input/mask lengths were both 20. Raw action was
`[-0.0346324,-0.0137920,-0.397059,-0.0001744,0.0447920,0.0180684,0.0]`; after the frozen LIBERO gripper adapter the
last component was +1.0.

Passive allocated/reserved VRAM was 4,720,663,552/4,836,032,512 bytes; peak was
5,060,689,920/5,196,742,656 bytes. Process wall was 9.33 s, maximum RSS 5,826,412 KiB, and RSS at result was
1,882,896 KiB. The bitsandbytes PyTorch-internal FutureWarning is non-semantic for this run. Canonical result:
`results/phase3a_openvla_4bit_single_image_v4.yaml`; result JSON SHA-256
`28e5a5e7ee41307d18bdf17223b81270e8b3d44bafad9c8fce156f76cae47063`; raw log SHA-256
`3c038aebd225cb0cb90d4f12266b990e9be4e1b6527ac600a2bcb769b0285620`.

This establishes 4-bit load and one action-generation feasibility only. The 1.024 s cold observation corresponds to
about 0.976 predictions/s, not a steady-state distribution, environment throughput or 20 Hz control. No environment
step occurred, so success, reward, termination and closed-loop reliability remain unknown; QLoRA/training also remain
unauthorized and untested. Next: design a dependency-isolated reset plus exactly one OpenVLA action/environment step,
then stop before an episode rollout.

## 26. Conversation close and continuation index (latest; 2026-08-27)

The short new-conversation entry is
`docs/handoffs/2026-08-27_phase3a_openvla_single_image_complete_resume.md`. It summarizes repository/authorization
boundaries, frozen Phase 1/2 evidence, the complete Phase 3A v1--v4 compatibility chain, the v4 PASS measurements,
the next dependency-isolated one-step bridge, and a verbatim Chinese message for starting a new conversation. This
file remains the complete cumulative historical fact source and must be read first from beginning to end. No training,
evaluation, download, installation, commit, push, upload or deletion was started while closing this conversation.

## 27. Phase 3A dependency-isolated single-step bridge static PASS (latest; 2026-08-27)

The user authorized project-local bridge development while retaining host execution for the real model/environment
gate. Static validation is PASS. New assets separate LIBERO ownership in `evla-core` from NF4 OpenVLA ownership in
`evla-openvla`, using a versioned local subprocess/JSON/PNG protocol with request/response hashes, strict identity,
shape, finite/range/gripper checks, a 180-second timeout, no-overwrite outputs and `finally` environment cleanup.

The online agent-view image is explicitly rotated 180 degrees to match upstream OpenVLA LIBERO evaluation, then uses
the frozen 90% center crop. The frozen NF4, local inference classes, force-hooks shim, token/mask alignment and
normalize/binarize/invert gripper path are unchanged. The static checker invoked the model worker's request-only path
with CUDA disabled and confirmed `environment_constructed=false`, `model_loaded=false`, `environment_steps=0`.
Canonical result: `results/phase3a_openvla_single_step_bridge_static.yaml`.

The pending host experiment is `phase3a_openvla_libero_single_step_bridge_v1_init0_seed20260827`, fixed to
`turn_on_the_stove`, init state 0 and seed 20260827. It permits exactly one reset, one model prediction and one
environment step, then stops. No episode, rollout set or QLoRA is authorized by a PASS. No formal execution, commit,
push, installation, download, upload or deletion occurred in this static update.

## 28. Phase 3A dependency-isolated online environment step PASS (latest; 2026-08-27)

`phase3a_openvla_libero_single_step_bridge_v1_init0_seed20260827` is PASS. The frozen init state 0 environment was
reset once, one hash-bound rotated online image crossed from `evla-core` to the isolated `evla-openvla` worker, one
finite in-range 7D action returned, and LIBERO accepted exactly one step before confirmed environment cleanup.

Model load/inference were 6.893/0.959 s; the environment step took 0.0104 s. Passive allocated/reserved VRAM was
4,720,663,552/4,836,032,512 bytes and peak was 5,060,688,896/5,196,742,656 bytes. Internal end-to-end time was
19.452 s, process wall was 26.56 s and maximum RSS was 5,829,496 KiB. The raw action was
`[0.0009769,0.2344853,0.0147059,-0.0113004,-0.0004727,-0.0000296,0.9960784]`; the adapted gripper was -1.0.

The first-step reward was 0 and success/terminated/truncated were false. This is not a failed episode: the gate proves
only interface feasibility. It provides no episode success rate, reward distribution, steady-state latency,
closed-loop reliability, loss, offline validation, gradients, optimizer throughput, training or QLoRA evidence.
All output hashes are frozen in `results/phase3a_openvla_libero_single_step_bridge_v1.yaml`; raw log SHA-256 is
`cb95cbfac70a77bb2ddd5f9ff1bea28da958d7990e496adec80091e87d637e35`. No network traffic, training, installation,
download, commit, push, upload or deletion occurred. Next: develop and statically verify a persistent-worker single
episode gate; do not run a batch rollout or QLoRA.

## 29. Phase 3A persistent-worker single-episode static PASS (latest; 2026-08-27)

The next gate has been implemented but not formally executed. Static validation is PASS for a new-ID single episode
on development init state 0. `evla-openvla` loads the model once and serves hash-bound per-step image requests while
`evla-core` owns the environment. The coordinator stops on termination/truncation or 600 steps, flushes trajectory
JSONL evidence, records latency/resource/action/reward fields, and guarantees worker reap plus environment close.

The checker ran the worker's request-only path with CUDA disabled: no model load, environment construction or episode
occurred. Canonical result: `results/phase3a_openvla_single_episode_static.yaml`. Pending formal ID:
`phase3a_openvla_libero_single_episode_v1_init0_seed20260827`. This is one development episode only; regardless of
success, it does not authorize batch rollout, held-out evaluation, QLoRA or training. No commit or push occurred.

## 30. Phase 3A first complete OpenVLA episode success (latest; 2026-08-27)

`phase3a_openvla_libero_single_episode_v1_init0_seed20260827` is PASS for execution and policy success. The frozen
official checkpoint completed `turn_on_the_stove` on development init state 0 in 85 steps, sum/max reward 1.0,
terminated true and truncated false. The persistent worker served exactly 85 predictions and shut down cleanly;
environment and worker cleanup were confirmed.

Model load was 5.781 s. Inference mean/median/P95 were 0.4214/0.4178/0.4367 s, total 35.815 s and pure throughput
2.37 predictions/s. Peak allocated/reserved VRAM was 5,061,286,912/5,196,742,656 bytes. Internal episode time was
54.549 s; process wall was 59.08 s and maximum RSS was 5,827,332 KiB. There were no swaps or network messages.

This N=1 development success proves closed-loop task-completion feasibility only; it is not a success-rate estimate.
No training/loss/offline-validation/gradient/checkpoint metric applies. Canonical result:
`results/phase3a_openvla_libero_single_episode_v1.yaml`; trajectory/raw-log hashes are
`084df2716c41e11f47b4c0b1e9db8915aadb35641e961470ea231d6bcb3d28df` and
`bdba26073a6ca03db407a41d8a17e9addb7dd68f2579e3633b599517cfbf44fa`. Next: bounded development init states
0--4 under one new ID; no held-out final, checkpoint selection or QLoRA is yet authorized by this result.

## 31. Phase 3A bounded dev5 rollout static PASS (latest; 2026-08-27)

The frozen development rollout entrypoint is implemented and static PASS. New ID
`phase3a_openvla_libero_dev5_v1_init0to4_seed20260827` covers exactly init states 0--4 with seeds
20260827--20260831, one persistent OpenVLA load, five independent episodes and a 600-step cap each. Execution PASS
is kept separate from per-episode policy success; trajectory and IPC evidence are flushed incrementally and cleanup
reaps the worker and closes the environment. No CUDA/model/environment work occurred in the static checker.
Canonical static result: `results/phase3a_openvla_dev5_static.yaml`. No held-out state, final evaluation, checkpoint
selection, QLoRA, training, commit or push occurred.

## 32. Phase 3A bounded OpenVLA dev5 rollout complete (latest; 2026-08-27)

`phase3a_openvla_libero_dev5_v1_init0to4_seed20260827` is PASS and achieved strict success on all five frozen
development init states 0--4. Completion steps were 85/87/78/86/85; all sum/max rewards were 1.0. There were 421
environment steps/predictions. Inference median/P95 were 0.4125/0.4386 s, pure throughput 2.42 predictions/s, peak
allocated/reserved VRAM 5,061,286,912/5,196,742,656 bytes, process wall 211.38 s and max RSS 5,835,520 KiB.

The worker loaded once, returned 421 valid actions, shut down cleanly, and the environment closed. This is 5/5
development evidence with Wilson 95% approximately 56.6%--100%, not a final 100% claim. No loss, validation,
gradients, checkpoint creation or training applies. Canonical result: `results/phase3a_openvla_libero_dev5_v1.yaml`;
trajectory/raw-log hashes are `32e5a253e44e850b8412a4913047e5f2c74472f4c0e79e733c55d3c8b75c1b17` and
`94a4e6a1fd71987e0d9b457bbfe243ba4250d512bb77dbb1f611df5c9f415e06`.

The only untouched init states are 5--49. Preregister them as the base-policy held-out final before running them or
beginning QLoRA design. No held-out execution, training, commit or push occurred in this update.

## 33. Phase 3A base-policy held-out final preregistered (latest; 2026-08-27)

The untouched init states 5--49 are now frozen as the 45-episode base-policy final under
`phase3a_openvla_libero_final_heldout45_v1_init5to49_seed20260901`, with evaluator seeds 20260901--20260945,
one persistent model load and a 600-step horizon. Development states 0--4 are explicitly excluded. Static validation
PASSed without constructing an environment or loading the model. Canonical static result:
`results/phase3a_openvla_final45_static.yaml`. Once executed, final outcomes cannot revise this protocol. No held-out
episode, QLoRA, training, commit or push occurred in this static update.

## 34. Phase 3A OpenVLA base-policy held-out final complete (latest; 2026-08-27)

`phase3a_openvla_libero_final_heldout45_v1_init5to49_seed20260901` is PASS. The frozen official NF4 checkpoint
achieved 45/45 strict successes on every init state excluded from OpenVLA development, indices 5--49; all sum/max
rewards were 1.0. Mean/median steps were 91.87/76, range 65--593 and P95 101.4. State 12 needed 593/600 steps and
state 40 needed 221, so success is universal on the finite set but efficiency/horizon margin is not uniform.

The run used 4,134 predictions. Inference mean/median/P95 were 0.4097/0.4090/0.4325 s, pure throughput 2.44
predictions/s, and internal rollout 1,839.01 s. Peak allocated/reserved VRAM was
5,061,286,912/5,196,742,656 bytes; process wall was 1,844.55 s and maximum RSS 5,768,988 KiB. Worker/environment
cleanup PASSed with zero swap and socket traffic.

This exactly enumerates the frozen 45-state held-out finite set. Bernoulli Wilson 95% is approximately
92.13%--100%. The result remains single-task, fixed-runtime and unperturbed-vision evidence. It contains no loss,
offline validation, gradients, optimizer, checkpoint creation or training. Canonical result:
`results/phase3a_openvla_libero_final_heldout45_v1.yaml`; trajectory/raw-log hashes are
`4c236d4ad942e8e954cd21dcac80962fa906c2c3ab1f4a48f178c3fd120f6d29` and
`b195f1b94f0273b96d77cbc50d871b85a6409d6a21740a38278480ce54b06873`.

Phase 3A base-policy feasibility is complete. Next: Phase 3B no-install QLoRA source/data contract, followed only
after review by a batch-1 forward/backward resource gate. No QLoRA, training, commit or push occurred here.

## 35. Phase 3B QLoRA source/data contract started (latest; 2026-08-27)

The no-install source audit is PASS. Upstream pins PEFT 0.11.1, quantized preparation, LoRA all-linear targets and
action-token-only supervision. The first project resource gate freezes rank 8, alpha 8, batch 1 and exactly one
forward/backward with no optimizer step. It will use a locally exported real LIBERO sample and upstream gripper
training transform `1-clip(raw,0,1)`, without TensorFlow/RLDS/dlimp, TRL, datasets or W&B.

Current `evla-openvla` has none of PEFT/TRL/datasets installed. Only `peft==0.11.1` is required and must be added by
a separate host environment gate using `--no-deps`; the checker imports PEFT without CUDA/model work. Canonical
contract: `results/phase3b_openvla_qlora_static_contract.yaml` and
`docs/phases/phase_3b_openvla_qlora.md`. No installation, forward, backward or training occurred yet.

## 36. Phase 3B PEFT runtime and QLoRA resource static gates PASS (latest; 2026-08-27)

The exact no-deps installation of PEFT 0.11.1 is PASS; Torch/Transformers/Accelerate remained
2.11.0+cu128/4.40.1/1.10.1 and `pip check` found no broken requirements. The import gate used no CUDA or model.

A real `demo_0/t=0` fixture is hash-frozen from the verified LIBERO HDF5. Raw gripper -1 becomes training gripper
+1 under upstream `1-clip(raw,0,1)`. The formal resource gate is frozen as rank 8, alpha 8, all-linear, batch 1,
seven supervised action tokens, one forward/backward, zero optimizer steps and no checkpoint. Static validation
PASSed without CUDA/model execution. Canonical records: `results/phase3b_openvla_peft_runtime_gate.yaml` and
`results/phase3b_openvla_qlora_resource_static.yaml`. Formal GPU resource execution remains pending; no training run
has started.

## 37. Phase 3B QLoRA resource v1 writer failure and v2 fix (latest; 2026-08-27)

V1 is FAIL. It loaded all shards and reached beyond forward, finite loss, metric alignment, backward and finite
gradient checks, but the path variable `output` was shadowed by the model output. Result writing then called
`.mkdir()` on `PrismaticCausalLMOutputWithPast`. This is not OOM; however, quantitative loss/grad/VRAM values were
not serialized and cannot support PASS. Process wall was 9.34 s, max RSS 5,775,856 KiB, exit 1. Canonical failure:
`results/phase3b_openvla_qlora_resource_v1.yaml`.

V2 uses a new ID and changes only `output_dir` versus `model_output` variable names. A no-CUDA writer regression is
PASS; all model/data/LoRA/compute contracts remain identical. Canonical static record:
`results/phase3b_openvla_qlora_resource_v2_static.yaml`. No optimizer step, checkpoint or training occurred.

## 38. Phase 3B QLoRA resource v2 forward/backward PASS (latest; 2026-08-27)

V2 formal execution is PASS on exactly one frozen real sample. Rank-8 all-linear LoRA exposes 27,707,072 trainable
parameters (0.7048% of 3,931,114,624 with adapters). Loss is 17.8423, action-token accuracy 0/7, continuous-action
L1 0.4807 and gradient norm 539.24; all gradients are finite. These are untrained one-sample diagnostics, not an
offline validation estimate, environment reward/success or training trend. Forward/backward took 0.622/0.463 s.
Peak allocated/reserved VRAM was 6,621,465,600/6,851,395,584 bytes; host wall was 8.44 s, max RSS 5,835,192 KiB,
with zero swap. Exit was 0. No optimizer step or checkpoint occurred. Canonical result:
`results/phase3b_openvla_qlora_resource_v2.yaml`. Next is exactly one clipped optimizer-step resource gate under a
new ID; formal multi-step training remains outside the verified boundary.

## 39. Phase 3B optimizer gate v1 sentinel failure; v2 static fix (latest; 2026-08-27)

Optimizer gate v1 is FAIL after reaching `optimizer.step()`: it asserted a delta on only the first trainable LoRA
parameter, which can legitimately have zero first-step gradient/update under Gaussian LoRA initialization. It is not
OOM; wall was 15.57 s, max RSS 5,808,432 KiB, zero swap and exit 1. No result was serialized, so loss, gradient,
optimizer state, VRAM peak and delta are unavailable. V1 log/source are frozen in
`results/phase3b_openvla_qlora_optimizer_v1.yaml`. V2 keeps the compute/optimizer contract and selects the trainable
parameter with the largest finite nonzero pre-clip gradient as its update sentinel. Its no-CUDA static regression is
PASS; formal V2 execution is pending.

## 40. Phase 3B optimizer gate v2 PASS (latest; 2026-08-27)

V2 is PASS for exactly one real sample and one clipped AdamW update. Loss 17.8423, action-token accuracy 0/7 and
continuous L1 0.4807 reproduce the untrained fixture baseline. Pre-clip gradient norm 539.247 was finite; clipping
to 1.0 completed. AdamW state is 218,256,208 bytes, step time 0.176 s, and a nonzero-gradient LoRA-B sentinel changed
by max 1.0e-4/L2 0.01810. Peak allocated/reserved VRAM was 6,621,465,600/6,834,618,368 bytes; wall 10.94 s,
max RSS 5,774,996 KiB and swap zero. No checkpoint was created. This is one-sample resource feasibility, not a
training curve, offline validation, reward, success or policy improvement. Canonical result:
`results/phase3b_openvla_qlora_optimizer_v2.yaml`. Next freeze the deterministic 40/10 demonstration training and
validation contract before a short multi-step training smoke.

## 41. Phase 3B deterministic train20/val10 checkpoint smoke static PASS (latest; 2026-08-27)

The frozen 40/10 demonstration split has 3,571/889 frames. Dependency isolation is preserved: `evla-core` exported
20 seeded distinct training PNG/action pairs plus one fixed timestep-0 sample from every validation episode; the
`evla-openvla` process needs no HDF5 installation. The fixture manifest SHA-256 is
`930f19982eff640a8450ca43fa939149ae643927d99393b1c066f1c9755f70db`. The new smoke runs validation at step 0 and
20, exactly 20 clipped AdamW updates, then writes adapter, optimizer/RNG, resolved config and per-file hashes. Static
PASS used no CUDA/model. The val10 result is plumbing-only, not the complete 889-frame offline distribution.

## 42. Phase 3B train smoke v1 silent non-execution; v2 entrypoint fix (latest; 2026-08-27)

Formal v1 is FAIL even though exit status was 0. The module defined `main()` but omitted the `__main__` invocation;
therefore model load, validation, training and checkpoint creation were all zero. Wall was 1.33 s, max RSS 688,484
KiB and swap zero. Raw log/source are frozen in `results/phase3b_openvla_qlora_train_smoke_v1.yaml`. V2 uses a new
ID and explicit wrapper call while preserving all model/data/optimizer contracts. Its no-CUDA entrypoint regression
is PASS; formal v2 is pending.

## 43. Phase 3B train smoke v2 OOM and v3 memory fix (latest; 2026-08-27)

V2 is FAIL at the first training forward after initial val10. It requested 252 MiB with 99.44 MiB free; PyTorch
reported 6.08 GiB allocated and 201.87 MiB reserved/unallocated. No optimizer step, output or checkpoint occurred;
initial validation was not serialized. Wall 13.01 s, max RSS 5,825,552 KiB, zero swap. V3 explicitly deletes
validation logits/tensors, runs GC and empties CUDA cache at phase boundaries, and requests expandable allocator
segments. Static PASS; model/data/optimizer contracts are unchanged and formal v3 is pending.

## 44. Protocol-invalid v2 rerun completed compute (latest; 2026-08-27)

The pasted command was v2, not v3. Because the project-authored `test ! -e` guards lacked shell errexit, execution
continued and overwrote the prior v2 raw log. Compute itself PASSed 20 finite clipped updates and produced a hash-
verified, CPU-loadable adapter/optimizer/RNG checkpoint. Train first/last loss was 22.7779/12.4246; first/last-10
means 14.9248/10.3290. Val10 loss changed 16.9151 to 8.8129 and token accuracy 4/70 to 8/70. Peak allocated/reserved
VRAM was 6,866,165,248/7,291,797,504 bytes; wall 26.76 s and max RSS 5,774,656 KiB. However reused ID and overwritten
raw log make the formal classification FAIL. The checkpoint is quarantined from resume/selection/evaluation. V3 is
still pending and commands must use `set -euo pipefail` plus explicit existence guards.

## 45. Phase 3B canonical train20/val10 checkpoint smoke v3 PASS (latest; 2026-08-27)

V3 is the first clean canonical training smoke PASS. Preflight GPU use was 733 MiB. All 20 optimizer steps and finite
clipped gradients completed. Train first/last loss 22.7779/12.3476; first/last-10 mean 14.9103/10.3167. Val10 loss
16.9151→8.8018 and token accuracy 4/70→9/70. Peak allocated/reserved VRAM 6,755,235,328/7,128,219,648 bytes;
train+validation 18.55 s, wall 26.99 s, max RSS 5,778,836 KiB, zero swap. Five checkpoint files hash-verified and
optimizer state CPU-loads at step 20. Canonical records: `results/phase3b_openvla_qlora_train_smoke_v3.yaml` and its
checkpoint manifest. This is not environment success/reward or full offline validation. Next is all 889 frozen
validation frames; checkpoint remains ineligible for resume/selection/environment evaluation until that gate.

## 46. Phase 3B fullval889 static gate PASS (latest; 2026-08-27)

All 889 frames from the frozen 10 validation demonstrations were exported by `evla-core` into a hash-verified
PNG/JSON manifest (`344f1d7b...e1378`). The `evla-openvla` evaluator will load the canonical v3 step-20 adapter
read-only and report loss, token accuracy, continuous L1, episode metrics, time and memory. Static PASS used no CUDA.
Formal execution is pending; it performs no training, environment rollout or checkpoint selection.

## 47. Step-20 fullval889 PASS; paired base gate static PASS (latest; 2026-08-27)

Step-20 adapter evaluated all 889 frames/6,223 tokens: mean loss 8.9271, accuracy 19.97%, mean continuous L1 0.2073.
Validation was 206.17 s at 4.31 frames/s; peak allocated/reserved VRAM 5,403,944,960/5,567,938,560 bytes; wall
213.96 s, max RSS 5,783,256 KiB, no swap. This does not establish improvement without a paired base run. The exact
same-fixture base evaluator is static PASS and pending. No reward/success, selection or training occurred.

## 48. Base fullval v1 direct-model path FAIL; v2 static fix (latest; 2026-08-27)

Base v1 loaded all shards then FAILed before frame 1 because metric alignment used the PEFT-only
`model.base_model.model` path on direct OpenVLA. No offline metric or output exists; wall 12.21 s, max RSS 5,764,460
KiB, zero swap. Shell errexit caused the terminal window to close after Python exit 1; GPU/OS did not crash. V2 uses
`wrapped=model`, a new ID/log/output, and passes static regression. Formal v2 remains pending.

## 49. Paired base versus step-20 fullval889 offline improvement PASS (latest; 2026-08-27)

Base v2 is PASS on the identical 889 frames: loss 16.9597, token accuracy 600/6,223 (9.64%), continuous L1 0.3077.
Step-20 changes are loss -8.0326/-47.36%, accuracy +10.33 percentage points/+643 tokens, and L1 -0.1004/-32.63%.
Loss improves on 883/889 frames, L1 on 732/889, and all three metrics improve in every one of 10 episodes. Canonical
paired record: `results/phase3b_openvla_step20_vs_base_fullval889_paired.yaml`. Offline improvement is PASS and the
checkpoint is eligible only for a development environment gate. No fine-tuned reward/success, final selection or
held-out environment authorization exists.

## 50. Step-20 adapter single development episode PASS; dev5 static PASS (latest; 2026-08-27)

Adapter init0 execution and policy success both PASS at step 102/reward 1/terminated, versus base step 85. Median
inference 0.4540 s, peak allocated/reserved VRAM 5,403,944,960/5,567,938,560 bytes, episode 65.17 s, host wall
71.85 s, max RSS 5,847,248 KiB, no swap. The +17 steps/+20% at n=1 is not stable efficiency evidence. The adapter-
aware init0--4 dev5 gate is static PASS and pending; held-out remains unauthorized.

## 51. Adapter dev5 v1 GPU-contention FAIL; v2 preflight static PASS (latest; 2026-08-27)

V1 loaded base shards but adapter load OOMed requesting 2 MiB with only 23.94 MiB GPU free. Worker never became
ready and zero episodes started; this is not policy failure. Wall 15.45 s, max RSS 5,768,284 KiB, zero swap. The
partial output contains only resolved config. V2 is a new ID with unchanged compute contract and a host preflight
requiring at least 8,000 MiB free before launch. Static PASS; formal v2 pending.

## 52. Adapter dev5 v2 PASS; heldout45 final frozen (latest; 2026-08-27)

Dev5 v2 execution and success are 5/5. Adapter steps [102,77,73,71,72] total 395/mean79.0 versus base
[85,87,78,86,85] total421/mean84.2: aggregate -6.18%, faster on 4/5 but init0 slower by17. This is heterogeneous
development evidence, not superiority. Combined with paired offline improvement, step-20 is frozen for exactly one
init5--49 final. Final config/static checker PASS; no further tuning/checkpoint changes are allowed before or after
viewing held-out outcomes.

## 53. Phase 3B immutable heldout45 final and completion (latest; 2026-08-27)

Final execution PASS, policy success 43/45 (95.56%, Wilson 95% 85.17%--98.77%). Init12 and init44 fail at the
600-step cap; base succeeded them at 593 and 68. Adapter capped mean/median/P95 95.24/70/123.4 versus base
91.87/76/101.4. On 43 common successes adapter is faster/slower/tied on 32/10/1 with mean/median delta -9/-6.
Therefore typical successful control improves but catastrophic tails regress reliability from 45/45 to 43/45.
Phase 3B is complete: QLoRA engineering and offline adaptation PASS; environment non-regression FAIL. Frozen base
remains the reliable demo/deployment recommendation, adapter is research-only, and post-heldout tuning/rerun is
forbidden. Canonical final: `results/phase3b_openvla_qlora_step20_final_heldout45_v1.yaml`.

## 54. Latest resume entry created after Phase 3B completion (2026-08-27)

The newest conversation entry is
`docs/handoffs/2026-08-27_phase3b_complete_total_project_resume.md`. It summarizes all completed phases, the complete
Phase 3B failure/result chain, immutable boundaries, demo/resume assets and prioritized next work. New conversations
must read this cumulative handoff first, then the Phase 3A resume, then the Phase 3B complete resume. No new training,
evaluation, commit or push occurred in creating that documentation.

## 55. P0 MVP offline presentation implementation (latest; 2026-08-27)

P0 implementation started without multi-agent, environment rollout, CUDA inference, download or installation. A
read-only inventory found complete per-step PNG/JSON evidence for the OpenVLA base and QLoRA held-out finals, but
the ACT final retained only metrics/eval-info and no image frames. Therefore no ACT video was fabricated from
unrelated demonstration images.

The canonical offline build is `outputs/presentation/p0_mvp_v2_20260827`. It contains a frozen-base init5 success
video, a paired init40 typical-speedup video, a paired init44 base-success/QLoRA-tail-failure video, paired offline
metric and held-out reliability figures, and an architecture diagram. All are derived from frozen evidence with
zero environment steps/model inference and are hashed in `manifest.yaml`. Videos are 1280x720 H.264 at 10 display
fps; this is not controller frequency or inference throughput. The first v1 build is preserved as noncanonical
visual-QA history because its source images were not enlarged and comparison titles overlapped.

`scripts/verify_p0_assets.py` verifies source/output hashes and ffprobe structure read-only. Project report,
portfolio and resume material were added/updated. A new immutable ACT presentation capture contract is prepared at
`configs/eval/act_libero_p0_demo_init05_v1.yaml` with `scripts/record_p0_act_libero_demo.py`. It is exactly one
headless recapture of the already-frozen step-100k policy on held-out init5, with new output/log/result paths and no
claim-changing authority. Formal host capture is pending and must not run automatically.

## 56. P0 ACT capture and MVP completion (latest; 2026-08-27)

`p0_act_libero_ckpt100000_demo_init05_v1_seed20260901` is PASS for execution and policy success. The runtime
verified and strictly loaded the frozen step-100k checkpoint, then recaptured already-finalized held-out init5 in
83 steps with strict success and sum reward 1.0. This is presentation evidence only and does not revise the frozen
45/45 final or checkpoint selection.

There were 17 ACT policy forwards and 0.5748 s total timed inference (about 33.8 ms/forward including cold-start
effects). Peak allocated/reserved VRAM was 162,552,832/216,006,656 bytes; process wall was 21.44 s, maximum RSS
3,826,808 KiB, swaps/network zero. Training loss, offline validation, gradients, optimizer state and checkpoint
creation are not applicable. The robosuite private-macro warning was nonsemantic; the asset helper confirmed the
existing local cache and no download/network traffic occurred.

The 83-frame 1280x720 H.264 video, trajectory, config and raw log hashes are frozen in
`results/p0_act_libero_ckpt100000_demo_init05_v1.yaml`. Visual QA found the original capture-v1 title clipped, so
that file remains immutable/noncanonical. A layout-only offline recomposition under a new v2 output ID PASSed with
zero environment steps and model inferences. Its SHA-256 is
`7aea923245fc2f3b74f47fb3acd7f27d52a3b8445856b52d1470811f1936408a`.

P0 is now COMPLETE. `results/p0_mvp_v2_manifest.yaml` is the canonical portfolio manifest and the read-only verifier
checks all source/media hashes plus H.264 structure. No training, held-out rerun, QLoRA tuning, commit or push
occurred. The next roadmap item is P1 viewpoint robustness, but it requires a new uncontaminated protocol and must
not start automatically.

## 58. P0-HD ACT resolution gate PASS and dual-environment correction (latest; 2026-08-28)

The first ACT 640px capture execution and policy success are PASS: init5 completed in 82 steps/reward 1.0. It used
17 forwards, 1.0195 s total inference, 162,552,832/216,006,656-byte peak allocated/reserved VRAM, 24.45 s host wall
and 3,913,920 KiB maximum RSS, with zero swap/network. The 1920x1080 video visual QA PASSed and is materially sharper.

However, its policy pixels came from 640 rendering downsampled to 128, not the native 128 renderer; the one-step
difference from the prior presentation capture reinforces that this must remain a noncanonical resolution gate.
For canonical HD captures, policy/physics now remain in the native 128 environment while a second render-only 640
environment copies the exact MuJoCo state each step. The generic OpenVLA coordinator and five base/QLoRA init5/40/44
configs are static PASS under this design. The first formal gate is frozen base init5. No final evidence is revised.

## 57. P0-HD 640px presentation upgrade started (latest; 2026-08-28)

The user approved replacing 128px presentation enlargement with genuine 640x640 LIBERO rendering. The new contract
separates display capture from frozen policy input: ACT rotates and deterministically downsamples 640px RGB to its
unchanged 128px input; OpenVLA will retain its frozen crop/model preprocessing. No AI super-resolution is used.

`configs/eval/p0_hd_act_libero_init05_v1.yaml` and `scripts/record_p0_hd_act_libero.py` are static PASS. They freeze
the step-100k checkpoint, held-out init5, 640px dual-camera environment observations, 128px ACT input, and a
1920x1080/20fps/CRF-16 H.264 presentation output under new no-overwrite paths. Static validation executed no CUDA or
environment. Formal ACT HD capture is pending; OpenVLA base/QLoRA HD coordinators follow only after this first
resolution-decoupling run. These recaptures cannot revise any final result.

## 59. P0-HD OpenVLA dual-EGL v1 infrastructure FAIL; single-EGL v2 ready (latest; 2026-08-28)

OpenVLA base init5 HD v1 did not reach policy execution. It stopped while constructing the two EGL-backed LIBERO
environments: only robosuite warnings and a resolved config were written; worker/model startup, environment steps,
success/reward, inference, resource/timing footer and video are all absent. `nvidia-smi` then reported driver
communication failure. Classify and preserve this as infrastructure `FAIL`, not OpenVLA policy failure, at
`results/p0_hd_openvla_base_init05_v1_failure.yaml`; do not reuse its run ID or output directory.

The replacement v2 removes the second environment. One native-128 LIBERO environment supplies unchanged policy
pixels and physics, and its same MuJoCo sim/offscreen EGL context performs an additional direct 640 camera render for
presentation. `scripts/record_p0_hd_openvla_libero.py` and
`configs/eval/p0_hd_openvla_base_init05_v2.yaml` pass static syntax/schema/single-environment/no-overwrite checks.
Formal execution is blocked until the host NVIDIA driver is restored and `nvidia-smi` succeeds. Frozen Phase 3
results and all no-retuning/no-final-rerun boundaries remain unchanged.

## 60. P0-HD NVIDIA recovery PASS with remaining GPU contention (latest; 2026-08-28)

The reboot restored NVIDIA-SMI (driver 580.173.02, reported CUDA 13.0), so the prior driver outage is cleared. This
check ran no OpenVLA model, environment, policy inference or checkpoint validation. A separate `unitree_rl` Python
compute process, PID 35905, still used 3381 MiB; total GPU allocation was 3990/12227 MiB and utilization 64%.
Consequently the single-EGL v2 base-init5 capture remains deferred until contention is cleared, without killing or
altering that unrelated workload. See `results/p0_hd_gpu_recovery_gate_20260828.yaml`.

## 61. P0-HD GPU contention cleared; base-init5 v2 eligible (latest; 2026-08-28)

The second NVIDIA-SMI preflight PASSed after the `unitree_rl` compute process exited: 620/12227 MiB GPU memory used,
11607 MiB derived free, 12% utilization and no compute process. This clears the resource boundary for exactly one
single-EGL OpenVLA frozen-base init5 presentation capture. No model, environment, policy or checkpoint was exercised
by the preflight itself. Structured record: `results/p0_hd_gpu_resource_gate_20260828_004105.yaml`.

## 62. P0-HD OpenVLA base init5 single-EGL v2 PASS (latest; 2026-08-28)

The corrected one-environment design PASSed execution and strict policy success: frozen OpenVLA base init5 reached
reward 1.0 in 83 steps/83 predictions. Load was 4.619 s; inference total/mean/median/P95 was
28.437/0.3426/0.3369/0.3434 s, peak allocated/reserved VRAM 5,061,286,912/5,196,742,656 bytes, host wall 61.67 s
and max RSS 5,805,596 KiB with zero swap/network. Frozen revision `fa5ae1e...baf` and four-shard loading matched;
worker cleanup PASSed.

The hashed 1920x1080 presentation and 1080x1080 clean videos are H.264/yuv420p/20fps/83-frame outputs derived from
direct 640 rendering while policy IPC remains native 128. Visual QA PASSed. No loss/offline/grad metric applies and
the Phase 3A 45/45 result is unchanged. A fresh single-EGL base-init40 v2 config is statically prepared as the next
half of the typical-speedup comparison.

## 63. P0-HD OpenVLA base init40 v2 PASS (latest; 2026-08-28)

The typical-speedup pair's frozen-base half PASSed at 221 steps/reward 1.0 with 221 predictions, matching the frozen
outcome. Load was 5.509 s; inference total/mean/median/P95 74.359/0.3365/0.3366/0.3434 s; peak allocated/reserved
VRAM 5,061,286,912/5,196,742,656 bytes; host wall 132.18 s and max RSS 5,775,076 KiB with zero swap/network.
Checkpoint revision/shards and worker/GPU cleanup PASSed. Both 221-frame HD videos and the trajectory pass hashes,
media probe and visual QA. The QLoRA init40 single-EGL v2 config is statically ready under a new ID, changing only
the frozen adapter worker. It remains presentation-only and cannot revise the adapter's frozen 43/45 conclusion.

## 64. P0-HD QLoRA init40 PASS and typical-speedup HD pair complete (latest; 2026-08-28)

QLoRA init40 PASSed strict success in the frozen 70 steps/reward 1.0, versus base 221. This is 151 fewer steps
(-68.33%, 3.16x ratio), while adapter inference is slower at mean/median/P95 0.4599/0.4528/0.4627 s versus base
0.3365/0.3366/0.3434 s. Frozen adapter SHA `6b4fce2e...735c` matched. Peak allocated/reserved VRAM was
5,468,073,984/5,584,715,776 bytes; host wall 63.53 s and max RSS 5,848,468 KiB with zero swap/network.

The 70-frame HD outputs pass hash/media/visual QA. Interpret this only as typical-state control-path acceleration;
the adapter remains research-only at frozen 43/45. A new-ID single-EGL base-init44 config is ready to begin the
base-success versus adapter-timeout tail comparison without tuning, checkpoint replacement or final rerun.

## 65. P0-HD OpenVLA base init44 v2 PASS (latest; 2026-08-28)

Frozen base init44 PASSed strict success at exactly 68 steps/reward 1.0. Load was 4.257 s; inference
total/mean/median/P95 23.425/0.3445/0.3390/0.3447 s; peak allocated/reserved VRAM
5,061,286,912/5,196,742,656 bytes; host wall 53.58 s and max RSS 5,776,412 KiB with zero swap/network. Revision,
four shards, cleanup and the two 68-frame HD media/hash/visual checks PASSed.

Only the frozen QLoRA init44 known 600-step timeout remains for formal P0-HD capture. Its fresh v2 contract changes
only the frozen adapter worker and exists solely to illustrate the already-finalized reliability tail. It cannot
authorize tuning, checkpoint replacement or a Phase 3B final rerun.

## 66. P0-HD COMPLETE after expected QLoRA init44 timeout (latest; 2026-08-28)

QLoRA init44 is `EXPECTED_EXCEPTION`: execution PASSed, but the frozen adapter reproduced its known 600-step
timeout with reward 0 and no success. Load was 5.189 s; inference total/mean/median/P95
291.498/0.4858/0.4843/0.5036 s; peak allocated/reserved VRAM 5,468,073,984/5,584,715,776 bytes; host wall 407.17 s
and max RSS 5,872,152 KiB with zero swap/network. Frozen base revision, adapter SHA, four shards, 600 predictions,
worker/GPU cleanup and all 600-frame media/hash/visual checks PASSed.

P0-HD is COMPLETE. Offline composition performed zero environment steps/inferences and produced two 1920x1080
side-by-side videos under `outputs/presentation/p0_hd_comparisons_v1_20260828`: init40 typical acceleration and
init44 reliability-tail failure. `results/p0_hd_comparisons_v1_manifest.yaml` plus the P0-HD entries in
`results/p0_mvp_v2_manifest.yaml` are canonical. Base 45/45, QLoRA 43/45, research-only adapter status and all
no-retuning/no-final-rerun rules remain unchanged.

## 67. P0-Breadth multi-task protocol static PASS (latest; 2026-08-28)

A new frozen-base breadth gate is ready to address the portfolio's single-task limitation without training or
post-outcome selection. Five LIBERO-Goal tasks `[0,2,3,6,9]` cover articulation, two placement modes, a compound
drawer-and-bowl instruction and object-to-container control. Each uses fixed init states `[0,1,2]`, for 15 episodes
and seeds 20261001--20261015. The evaluator loads frozen revision `fa5ae1e...baf` once, rebuilds/closes the environment
per task and records every episode/step through hashed native-128 IPC.

Static validation PASSed with zero CUDA/environment/predictions and fresh no-overwrite paths. This is not a complete
LIBERO benchmark and cannot revise the single-task 45/45 result. All tasks and failures remain in the aggregate;
post-outcome task removal, init replacement or tuning is forbidden.

## 68. P0-Breadth v2 expanded to 10 tasks / 30 episodes (latest; 2026-08-28)

The user requested all ten LIBERO-Goal tasks. The five-task v1 static plan was never executed and is retained as
superseded history. V2 freezes task IDs 0--9 and init states `[0,1,2]` for every task, giving 30 episodes/seeds
20261001--20261030. It uses one frozen OpenVLA-base load, per-task environment teardown and per-step hashed native-128
IPC under entirely new output/log/result paths.

All 10 metadata name/language pairs, episode arithmetic, syntax, checkpoint/no-overwrite contracts and fresh paths
static PASSed with zero CUDA, environment steps or predictions. It is catalog-wide task coverage but not the full
official benchmark because only three states per task are evaluated. No result may be removed or selectively rerun.

## 69. P0-Breadth 10-task frozen-base result: 20/30 and 10/10 coverage (latest; 2026-08-28)

V2 execution PASSed all 30 preregistered episodes in 51:06. Frozen OpenVLA base achieved 20/30 strict success
(Wilson 95% 48.78%--80.77%) and at least one success on every LIBERO-Goal task. Task IDs 0--9 scored
`[2,2,3,1,3,2,1,3,1,2]/3`; all ten failures remained as 600-step timeouts. T3 compound drawer+bowl, T6 cream
cheese-to-bowl and T8 bowl-to-plate were weakest at 1/3.

The run used 8,554 steps/predictions; inference mean/median/P95 was 0.3346/0.3331/0.3491 s, peak allocated/reserved
VRAM 5,068,574,720/5,196,742,656 bytes, host wall 3,066.95 s and max RSS 5,773,664 KiB with zero swap/network.
Checkpoint/shards and cleanup PASSed. This supports catalog-wide coverage on the fixed small set, not official full
benchmark performance. Next presentation work may recapture the first successful state per task in HD without
changing 20/30 or selectively rerunning failures.

The CPU-only quantitative summary is hash-manifested at `results/p0_breadth_summary_v1_manifest.yaml`. Ten HD demos
are frozen to the first successful state per task (`[0,1,0,2,0,0,1,0,0,0]`), not selected by later visual review.
`scripts/record_p0_breadth_hd_demos.py` is static PASS with all paths fresh and cannot modify the 20/30 aggregate.

## 70. P0-Breadth ten-task HD presentation COMPLETE (latest; 2026-08-28)

The frozen first-success recapture batch PASSed 10/10 and exactly reproduced source step counts, totalling 1,296
environment steps/predictions. Host exit was 0; wall time 785.02 s, max RSS 5,860,536 KiB, zero swap/network, and
GPU cleanup passed. Median/P95 inference ranges were 0.3315--0.3388/0.3363--0.3457 s; max allocated/reserved VRAM
was 5,068,574,720/5,196,742,656 bytes. All checkpoints, results, video hashes, codecs and frame counts verified.

Task 6's corrupted original presentation encode remains preserved and noncanonical; an offline recomposition from
its intact clean master passed start/middle/end visual QA. The canonical ten-task montage is 1920x1080, 20 fps and
64.8 s under `outputs/presentation/p0_breadth_10task_montage_v1_20260828`, with manifests
`results/p0_breadth_hd_10task_v1_manifest.yaml` and `results/p0_breadth_10task_montage_v1_manifest.yaml`.
P0-Breadth is COMPLETE; the selected success montage does not revise frozen 20/30 reliability or remove failures.

## 71. Phase 4A viewpoint camera contract static PASS (latest; 2026-08-28)

Phase 4 begins with a zero-policy camera gate. Frozen base and preprocessing remain unchanged; later paired runs
replace the single image source only (`agentview` versus `eye_in_hand`). Development tasks `[0,3,6,7,8]`, final
tasks `[1,2,4,5,9]`, init states `[0,1,2]` and 30+30 paired episodes are preregistered before eye-in-hand outcomes.
The 18-check static result PASSed with zero CUDA/reset/step/prediction. Next is one reset with dual 128 observation
and direct 640 render hashing; it makes no success/reward claim and cannot alter Phase 3/P0 evidence.

## 72. Phase 4A camera runtime v1 FAIL and v2 correction (latest; 2026-08-28)

V1 reset successfully then exited 1 because direct render used observation alias `eye_in_hand`, while MuJoCo names
the camera `robot0_eye_in_hand`. It performed zero steps/predictions and closed the environment; failed output/log
are preserved in a structured failure record. V2 changes only this name under fresh paths, preserves the frozen
checkpoint/splits/boundaries, and static PASSed nine checks with zero CUDA/reset/step/prediction. Subsequent command
uses a child shell to avoid propagating `set -e` into and closing the user's interactive terminal.

## 73. Phase 4A corrected dual-view runtime gate PASS (latest; 2026-08-28)

V2 performed one reset, zero steps/predictions and clean close with exit 0. Native/direct agent-versus-wrist MAE was
76.4615/77.4986; all four 128/640 RGB assets passed nonconstant, hash and visual checks. Host wall was 12.17 s, max
RSS 2,486,184 KiB and swap/network zero. No model metrics, success or reward apply. Camera feasibility is complete;
Phase 4B paired development rollout requires a new static contract before execution.

## 74. Phase 4B paired viewpoint development static PASS (latest; 2026-08-28)

The new gate freezes development tasks `[0,3,6,7,8]`, three init states and two single-image sources into 15 paired
states/30 episodes. Pair members share task/init/seed; every episode gets a fresh explicit-init environment and one
persistent canonical frozen-base worker serves all predictions. Static 14-check validation PASSed with zero
CUDA/reset/step/prediction. Final tasks `[1,2,4,5,9]` remain untouched. Phase 4A's unloaded revision typo is logged
as metadata-only correction; Phase 4B binds the actual Phase 3A/P0 revision. No tuning/selection/selective rerun is
permitted.

## 75. Phase 4B viewpoint development: execution PASS, direct substitution NO-GO (latest; 2026-08-28)

Thirty episodes completed: agentview 10/15 and 5/5 task coverage versus eye-in-hand 0/15 and 0/5 coverage. All 15
wrist episodes timed out at 600. Pair outcomes were 10 agent-only, 0 wrist-only and 5 neither (two-sided exact
p=0.001953). Total steps/predictions 13,663; median/P95 inference 0.3404/0.3483 s; peak allocated/reserved VRAM
5,068,574,720/5,196,742,656 bytes; host wall 4,966 s, max RSS 5,867,244 KiB, zero swap/network. Checkpoint/shards and
worker cleanup PASSed. Direct substitution is development NO-GO; untouched final tasks are not run without a newly
preregistered adaptation hypothesis.

The canonical Phase 4B quantitative PNG/SVG is visual-QA PASS and hash-manifested by
`results/phase4b_viewpoint_dev_summary_v2_manifest.yaml`; v1 remains noncanonical due label overlap. A read-only
verifier covers result/log/trajectory/figure integrity with zero added environment/model work.

## 76. Phase 4C wrist-adaptation hypothesis ready for training smoke (latest; 2026-08-28)

The existing single-task dataset provides 3,571 train and 889 validation native wrist frames under the unchanged
40/10 episode split. Image/action/hash checks and a deterministic wrist-only 20-train/10-val fixture PASSed. A
rank-8 NF4 QLoRA 20-update smoke reuses frozen Phase 3B training machinery; corrected static validation PASSed 12
checks after preserving an expected-trainer-hash-only static failure. This is development/offline pipeline evidence
only; no checkpoint selection, environment-success inference or final-task execution is permitted.

## 77. Phase 4C wrist QLoRA smoke engineering PASS, offline signal mixed (latest; 2026-08-28)

Twenty updates completed with finite losses/gradients and a complete 879-tensor adapter plus step-20 optimizer/RNG.
Val10 loss improved 32.29%, but token accuracy fell from 16/70 to 13/70, so no checkpoint is selected and no closed-
loop claim follows. Peak allocated/reserved VRAM was 6,866,078,208/7,065,305,088 bytes; host wall 31.96 s, max RSS
5,776,904 KiB, zero swap/network. Next gate must be frozen paired fullval889 on wrist images before any rollout.

## 78. Phase 4D wrist fullval889 paired static contract PASS (latest; 2026-08-28)

A complete 889-frame native-wrist fixture is hashed. Base and step-20 adapter share identical validation episodes,
labels, fixture and canonical revision through frozen evaluator wrappers. Static 11-check validation PASSed with no
CUDA/frame/environment work. Base must execute and be audited before adapter; all metrics remain offline-only and do
not authorize rollout or touch final tasks.

## 79. Phase 4D frozen-base wrist fullval889 PASS (latest; 2026-08-28)

Base completed 889 frames with loss 16.6834, token accuracy 13.0805% and L1 0.28796. Throughput was 4.419 frames/s;
peak allocated/reserved VRAM 5,167,181,312/5,335,154,688 bytes; host wall 209.32 s and max RSS 5,768,068 KiB.
All hashes/revision matched and no environment work occurred. Adapter same-fixture offline evaluation is now allowed;
these metrics remain distinct from success/reward.

## 80. Phase 4D wrist adapter fullval889 paired PASS (latest; 2026-08-28)

On the identical 889-frame/6,223-token wrist fixture, step-20 QLoRA improved base loss 16.6834→9.0479 (-45.77%),
token accuracy 13.0805%→20.8099% (+7.73 percentage points), and continuous L1 0.28796→0.19459 (-32.43%). All
10 validation episodes improved on all three metrics. Throughput fell 13.28%, median/P95 latency rose
15.69%/15.43%, and VRAM rose about 4.7–5.8%. Integrity PASSed with zero training/environment work.

Decision is `OFFLINE_ADAPTATION_PASS_ENVIRONMENT_RECOVERY_UNKNOWN`: offline metrics are not success/reward. The
adapter is only a development rollout candidate; final tasks remain untouched pending a new frozen environment gate.

## 81. Phase 4E task-7 wrist development rollout static PASS (latest; 2026-08-28)

Phase 4E freezes one adapter rollout on task 7 init `[0,1,2]` using the exact Phase 4B paired seeds and immutable
base wrist reference (0/3, all 600-step timeouts). Adapter decisions are preregistered as 0/3 NO-GO, 1–2/3 partial
recovery and 3/3 strong recovery. Static checks PASSed 12/12 with zero CUDA/environment/model work. This cannot
authorize final tasks, retuning, selective reruns or cross-task ego-view claims.

## 82. Phase 4E execution PASS, closed-loop recovery NO-GO (latest; 2026-08-28)

The fixed wrist adapter scored 0/3 on task 7 init `[0,1,2]`; every episode timed out at 600 with reward 0, yielding
three neither-success pairs versus base. All 1,800 actions were finite/in-range and worker/resource/artifact integrity
PASSed. Mean inference was 0.4876 s, 43.02% slower than matching base wrist episodes; peak reserved VRAM was
5,584,715,776 bytes. The offline adaptation PASS therefore did not translate to closed-loop recovery. Freeze this
NO-GO: no retuning, checkpoint switching, selective rerun or final-task execution.

## 83. Phase 4 viewpoint closeout COMPLETE / NO-GO (latest; 2026-08-28)

Phase 4 now has a single frozen evidence chain: camera pipeline PASS; direct wrist substitution 0/15 NO-GO;
wrist QLoRA offline adaptation PASS; task-7 closed-loop recovery 0/3 NO-GO. Final tasks stayed untouched. The
canonical closeout document, visual manifest and read-only verifier make this negative result portfolio-ready while
preserving the key limitation that teacher-forced improvement is not closed-loop viewpoint transfer.
The Phase 4 read-only verifier PASSes 17/17 checks over four manifests and seven figure artifacts.

## 84. P0 project report and unified reproduction closeout COMPLETE (latest; 2026-08-28)

A standalone project report and claim-to-evidence reproduction index now cover all frozen Phase 0–4 results,
limitations, canonical media and verifier entrypoints. The closeout manifest hashes four documents plus the featured
ten-task montage and Phase 4 evidence figure. Its read-only verifier PASSes 16/16 checks, including all 35 indexed
local references, with zero CUDA/environment/model work.

## 85. Phase 5A world-model data feasibility PASS (latest; 2026-08-28)

The existing 50-demo task-7 dataset supports episode-safe H=1/2/4/8 temporal windows under the frozen 40/10 split.
All image/action/state/reward/done alignment checks PASSed. Copy-frame validation MAE is 0.769→2.402 for agentview
and 6.278→24.594 for wrist from H1→H8, showing a much harder moving-camera prediction problem. This CPU-only gate
authorizes baseline implementation contracts, not model training or any learned/world-model claim.

## 86. Phase 5B small world-model architecture static PASS (latest; 2026-08-28)

Parameter-matched ~1.29M action-free and action-conditioned latent models PASSed 16/16 real-data CPU shape, range,
H8 rollout, finite-loss/gradient and action-path checks. Peak RSS was 968,836 KiB; no CUDA/training/environment work
occurred. Random-weight values are not performance. A short training smoke may be implemented next, but remains
unauthorized until its protocol and resource gate are frozen.

## 87. Phase 5C paired training-smoke contract static PASS (latest; 2026-08-29)

The next gate freezes 256/128 episode-safe H4 windows and paired 100-update action-free/conditioned runs with
identical data order and encoder/decoder initialization. H1–4 pre/post and copy-frame metrics plus both complete
checkpoints are mandatory. Static checks PASSed 14/14 with no CUDA/model/training/environment work. The smoke cannot
select a checkpoint or authorize formal training/planning claims.

## 88. Phase 5C smoke engineering PASS, offline baseline NO-GO (latest; 2026-08-29)

Both 100-update models optimized stably and produced complete checkpoints, but post-smoke H4 MAE
free/conditioned=11.526/11.722 versus copy-frame=1.732. Conditioned is worse than action-free at all H1–4. Freeze
`ENGINEERING_PASS_OFFLINE_BASELINE_NO_GO`: no checkpoint selection, formal training, planning or environment claim.

## 2026-08-29 continuation: Phase 5C–5E

Phase 5C established an engineering PASS but offline baseline NO-GO: after paired 100-update training, H4 pixel MAE
was 11.526 action-free and 11.722 action-conditioned versus copy-frame 1.732, so neither model nor action
conditioning was accepted. No checkpoint was selected.

Phase 5D introduced a new copy-preserving bounded-residual mechanism with exact-copy zero initialization and passed
12/12 CPU static checks. The original 40 training episodes were nested into 32 training and 8 development episodes;
the original 10 validation episodes are now prior-exposed confirmation evidence.

Phase 5E freezes a development-only paired smoke using 256/128 H4 windows from the 32/8 split, excluding all
confirmation episodes. Static v1 exposed a canonical-YAML packaging omission without GPU work; corrected static v2
passed 18/18. GPU execution remains host-pending; conditioned H4 must beat
both copy-current-frame and residual action-free to earn development GO, which still cannot authorize confirmation
evaluation or formal training automatically.

Phase 5E host execution then PASSed technically but returned `DEVELOPMENT_NO_GO_ACTION_CONDITIONING`. H4 MAE was
copy/free/conditioned 1.59125/1.59223/1.59303; conditioned was worse than both baselines. Both step-100 checkpoints
and optimizer states are complete, but neither is selected. Confirmation episodes, environment evaluation and
longer training remain forbidden. Phase 5 stops the current mechanism and the next project line is Phase 6 Feagine
bridge static interface and safety-contract work.

Current report/reproduction documents are indexed by `results/project_closeout_v2_phase5e_manifest.yaml`; the
2026-08-28 P0 v1 closeout remains an immutable historical snapshot.

Phase 6A then implemented a provisional single-section PCC software bridge and PASSed 14/14 static/safety checks
after preserving two packaging exceptions and one bad-fixture FAIL. 256 FK/IK round trips had maximum error
8.58e-16 m with zero CUDA/environment/hardware work. This is not an A03 model: actual MJCF, geometry, PCC state,
actuator mapping and control period are absent and required before Phase 6B simulation.

The project-local Feagine v1.0.1 distribution was subsequently found and Phase 6B static binding PASSed 19/19.
Actual A03 Type-2 facts are 3×0.16 m PCC sections, five samples/section, six theta/phi variables, 48 capsule proxy
joint/actuator pairs, grasper rotation plus two fingers, and a 0.005 s MJCF timestep. Package default `a03` is not
present; explicit `a03_type_2` is mandatory. Runtime packages remain uninstalled and no MJCF load/step occurred.

Phase 6C isolated runtime zero-step protocol is implemented and static 6/6 PASS. It will create a new environment,
build the supplied core and install the local wheel, then load/bind MJCF and exercise FK/IK/control writes without
calling `mj_step`; host execution is pending.

Phase 6C v1 installation/build succeeded; zero-step gate was 10/11 FAIL because XML static counting included one
unnamed default joint template. Runtime correctly reports 51 named joints and 51 actuators; every other check,
including core IK and time 0→0, passed. v1 is preserved. Count-only v2 recovery static PASSed 9/9 and reuses the
installed environment without reinstall/rebuild.

Phase 6C corrected v2 then PASSed 11/11: 51 joints/actuators, finite controls, core IK in 5 iterations at 74.67 μm
position error, and data time 0→0. Phase 6D one-second/200-step headless dynamics protocol is static 7/7 PASS and
host-pending; it carries no target-reaching, reward, success, policy or hardware claim.

Phase 6D v1 completed 200 steps but FAILed 6/7 on joint limits despite finite state, exact 1s time and 6.70cm tip
motion. Since v1 omitted joint-level penetration, identical-command diagnostic v2 is static 8/8 PASS and host-pending;
no threshold relaxation or target-reaching authorization exists yet.

Phase 6D v2 diagnostic subsequently executed PASS. Among 51 limited joints, only slide joint
`feagine_grasper_joint_1` crossed its 0.005 m upper limit, reaching 0.0050117734 m: 11.77 μm penetration, or 11.77%
of the preregistered 0.1 mm material threshold. No hinge penetration occurred. Decision is
`SOFT_LIMIT_PENETRATION_WITHIN_DIAGNOSTIC_THRESHOLD`; v1 FAIL remains immutable and target-reaching remains a
separate, not-yet-authorized gate.

Phase 6E then froze that first bounded target-reaching gate. Static validation PASSed 10/10 without MuJoCo import
or environment steps. It uses a core-generated small reachable target, 400 steps at 5 ms, target displacement
0.02–0.08 m, terminal error at most 15 mm, at least 80% error reduction and the Phase 6D material thresholds.
Host execution PASSed 9/9. The 53.896 mm initial target error fell to 9.504 mm after 400 steps (82.366% progress),
with four-iteration IK, zero hinge penetration, 1.29e-12 m slide penetration and zero contacts. The gate contains
no VLA, image, arbitrary workspace target, reward or hardware work.

Phase 6F froze four externally specified local deltas (±35 mm X/Y), independently reset and strictly 4/4. Runtime
returned 0/4 FAIL. ±X failed IK after one iteration and did not move. ±Y passed core IK in 16 iterations but MJCF
errors expanded from 35 mm to 241.63/271.86 mm. All states/times remained valid with no material limit violation or
contact, isolating a target→section/controller mapping failure. No target replacement, relaxation or VLA is allowed.

Phase 6G same-target mapping diagnostic is now static 10/10 PASS and host-pending. It changes no targets, budgets or
mapping, and only records raw/applied section angles, 48 PCC controls, predicted/actual displacement directions and
gains. Diagnostic completion cannot revise the immutable Phase 6F 0/4 FAIL.

Phase 6G runtime completed. ±X returned zero angles/controls from the straight-start IK singularity. ±Y raw IK
solutions contained negative theta values that the A03 preset clamped to zero, changing one/two sections. After
clamping, core-predicted versus MJCF-actual displacement cosine was 0.99941/0.99927, locating the primary defect
before the actuator/dynamics layer. Candidate signed-theta canonicalization and multi-start seeds remain unexecuted.

Phase 6H CPU-only gate validated signed-theta canonicalization on 260 FK cases (maximum position/direction difference
9.22e-8 m/4.28e-7), but returned 4/5 FAIL because none of eight frozen seeds solved any original target within
0.1 mm. Best residuals clustered at 1.592–1.615 mm. The fixed-height lateral targets are likely geometrically outside
the fixed-length reachable set; no tolerance relaxation or MJCF recovery is authorized.

Phase 6I scanned 0–5 mm axial retraction at 0.25 mm increments while preserving 35 mm lateral displacement,
direction and 0.1 mm IK tolerance. All ±X/±Y directions first became reachable at 1.50 mm, with zero spread across
directions. v1 preserved a 5/6 FAIL caused only by exact float-list comparison; v2 reused all 672 IK attempts and
PASSed 10/10 with tolerant grid validation and zero new IK calls. No MJCF recovery has run.

Phase 6J now freezes the same four 35 mm lateral directions with the measured 1.50 mm axial retraction, fixed
first-reachable seeds and signed-theta canonicalization. Static validation PASSed 11/11; host execution is pending.
Environment gates remain ≤15 mm, ≥70% progress and strict 4/4; Phase 6F 0/4 remains immutable.

Phase 6J runtime returned 2/4 FAIL. All four directions passed IK, terminal error (~9.30 mm) and progress (~73.44%),
so reachability and signed-theta corrections recovered motion. ±Y fully passed; ±X failed only because maximum slide
penetration 0.2631 mm exceeded the frozen 0.1 mm material threshold. Hinges and contacts were zero. Joint identity
and peak time were not recorded, so tolerance/controller changes remain forbidden pending same-target diagnosis.

Phase 6K same-target per-slide-joint diagnostic is static 10/10 PASS and host-pending. It preserves all Phase 6J
commands and records joint identity/range/extrema plus first and peak violation steps. It cannot revise 2/4 FAIL.

Phase 6K runtime diagnostic completed. +X violated `feagine_grasper_joint_0` and -X the mirrored `joint_1`; each
started at the fully-open -0.013 m lower bound and reached about -0.01326307 m at step 1. Opposite-finger penetration
was only ~2.45 μm and ±Y was ≤3.90e-11 m. This identifies a hard-stop startup transient, while Phase 6J remains 2/4.

Phase 6L gripper-margin scan executed normally but selected no safe command. Commands 0.00/0.01/0.02/0.05 were all
2/4; ±X penetration decreased monotonically from about 0.2631 to 0.1150 mm while ±Y passed and arm error/progress
held at about 9.30 mm/73.44%. The best preregistered command still exceeded the frozen 0.1 mm threshold by about
0.015 mm, so Phase 6L is execution PASS and selection NO-GO; Phase 6J remains 2/4.

On 2026-09-04 an explicitly authorized storage prune removed 550 regenerable optimizer/noncanonical checkpoint
payloads (26.080067 GiB), reducing the project from about 36 GiB to 9.4 GiB. All results, raw logs, failure evidence,
presentation media and canonical inference models remain. The pre-delete path/size/SHA-256 inventory is
`results/storage_prune_recommended_20260904_manifest.yaml`; pruned training states are no longer locally resumable.
