# Embodied VLA Lab handoff — PushT Phase 1C complete and reproducibility frozen

Date: 2026-08-26
Project root: `/home/kxj/embodied_ws/embodied_vla_lab`
Environment: `evla-core`
Branch: `main`
HEAD at handoff creation: `121ba50 docs: add reproducible PushT run profiles`

This is the primary continuation document. It consolidates the useful state from the project’s prior conversations,
older handoffs, tracked experiment records, raw-log provenance, and the current ACT/Diffusion session. Read this file
completely before proposing or executing further work.

## 1. User workflow and authorization boundary

The user normally runs long host commands manually and pastes the terminal result back. For every pasted result:

1. First classify it as `PASS`, `FAIL`, or `EXPECTED_EXCEPTION`.
2. Analyze loss, offline eval, gradient norm/clipping, throughput, GPU allocation, process memory, elapsed time,
   closed-loop reward/success, latency, and the boundary of what the result proves.
3. Update the relevant structured result YAML, checkpoint manifest, experiment README, hashes, and reproducibility
   assets.
4. Give exactly one next minimal and safe host command, or explicitly say that no further command is needed.

Do not use multi-agent work. Use the `evla-core` environment. Read-only checks of the same class are allowed.
Installation, dataset/model download, new training, GUI, W&B, Hub/upload, commit, and push each require separate,
explicit authorization. Evaluation and inference protocols should be preregistered before observing held-out results.
Never start training merely because a config or run profile exists.

Generated datasets, checkpoints, raw logs, eval outputs, and videos are intentionally Git-ignored. Track configs,
scripts, structured results, manifests, documentation, and SHA-256 identities instead.

## 2. Repository state and frozen dependencies

Recent commits:

```text
121ba50 docs: add reproducible PushT run profiles
30394f2 experiment: complete Diffusion Policy PushT baseline
3e323b6 eval: compare deterministic and VAE ACT on 100 seeds
a1c5778 experiment: complete deterministic ACT ablation
9a589d6 docs: finalize ACT PushT closed-loop diagnostics
```

Submodule pins:

```text
8b256a6c0d4769c3cc3e7e98f04940126398a391 lerobot
8f1084e3132a39270c3a13ebe37270a43ece2a01 libero
c8f03f48af692657d3060c19588038c7220e9af9 openvla
```

Core verified environment:

- Python 3.12.14
- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- LeRobot 0.6.2 editable pinned source
- gym-pusht 0.1.6 with the documented headless OpenCV workaround
- `diffusers==0.39.0`, matching pinned LeRobot’s `>=0.38.0,<0.40.0` constraint
- RTX 5070 Ti Laptop GPU, 12,227 MiB reported total memory
- MuJoCo EGL and LIBERO offline reset/step/dual-camera RGB runtime previously passed

Known expected exception: TorchCodec ABI loading fails on this environment and LeRobot falls back to explicitly
configured PyAV. This is accepted for the fixed PushT workflow. Do not “fix” it by changing the video backend or
package stack without a new scoped decision.

## 3. Fixed PushT data contract

Primary contract: `configs/dataset/pusht_v3.yaml`

- repo: `lerobot/pusht`
- revision: `7628202a2180972f291ba1bc6723834921e72c19`
- local root: `/home/kxj/.cache/huggingface/lerobot/lerobot/pusht`
- 206 episodes, 25,650 frames
- image shape: 96×96×3
- state/action shape: 2/2
- FPS: 10
- deterministic length-stratified episode split, seed 20260825
- train: 185 episodes, 23,043 frames
- validation: 21 episodes, 2,607 frames
- held-out environment development seeds: `20260825..20260829`
- independent paired seeds: `20260830..20260929` (100 seeds)

The exact validation episode list is in the dataset contract and all formal configs. Do not regenerate or alter the
split for comparisons to the frozen results.

## 4. Phase history before the current Diffusion work

### Phase 0 and Phase 1A

- CUDA/BF16, LeRobot, PushT headless reset/step/render, and ACT interface passed.
- PushT v3 dataset snapshot was downloaded only after authorization, audited, and pinned.
- Relevant files:
  - `scripts/check_phase0_core.py`
  - `scripts/check_phase1a_pusht_dataset.py`
  - `configs/dataset/pusht_v3.yaml`
  - `docs/phases/phase_00_environment.md`
  - `docs/phases/phase_01a_pusht_dataset.md`

### Phase 1B ACT progression

The sequence was deliberately gated:

1. Two-step ACT smoke.
2. Episode-51 overfit diagnostic.
3. Full-split batch-8 100-step resource gate.
4. Formal VAE ACT 100k training.
5. Fixed-data checkpoint reload/accuracy.
6. Fixed five-seed closed-loop checkpoint and replanning diagnostics.
7. Deterministic ACT ablation (`policy.use_vae=false`) at 30k.
8. Independent paired 100-seed comparison.

Formal VAE ACT config: `configs/train/act_pusht_phase1b_formal_100k.yaml`
Deterministic ACT config: `configs/train/act_pusht_phase1b_deterministic_30k.yaml`

Deterministic ablation kept dataset revision, 185/21 split, batch 8, BF16, chunk/action steps 50/10, no pretrained
backbone, optimizer, seed, 5k offline eval, and 10k/20k/30k checkpoints. Its core model change was
`policy.use_vae=false`.

Important ACT outcomes on the independent 100 paired seeds:

| Model | Success | Exact 95% CI | Mean sum reward | Mean max reward | Zero reward | Eval s/episode |
|---|---:|---:|---:|---:|---:|---:|
| deterministic ACT 30k | 6/100 | 2.23–12.60% | 72.3539 | 0.51557 | 7 | 1.168 |
| VAE ACT 25k | 2/100 | 0.24–7.04% | 67.3235 | 0.41226 | 22 | 1.394 |

Paired deterministic-versus-VAE success difference was not statistically stable: McNemar p=0.2891 and paired
bootstrap success difference interval `[-1pp,+10pp]`. Deterministic ACT did have a stable paired mean max-reward
improvement, interval `[0.02834,0.17757]`, and fewer catastrophic zero-reward episodes.

Key ACT records:

- `experiments/act_baseline/README.md`
- `results/phase1b_act_pusht_formal_100k.yaml`
- `results/phase1b_act_pusht_formal_100k_checkpoint_manifest.yaml`
- `results/phase1b_act_pusht_deterministic_30k.yaml`
- `results/phase1b_act_pusht_deterministic_30k_checkpoint_manifest.yaml`
- `results/phase1b_act_pusht_deterministic_checkpoint_selection_5ep.yaml`
- `results/phase1b_act_pusht_deterministic_vs_vae_100seed.yaml`

## 5. Phase 1C Diffusion Policy baseline

### Design and preflight

The goal was a strong same-data model-family baseline and the fastest credible path to a better PushT demo, not a
single-field ACT ablation. Config: `configs/train/diffusion_pusht_phase1c_baseline_30k.yaml`.

Fixed controls included dataset revision/split, batch 8, BF16, seed 20260825, no pretrained backbone, execute 10,
5k validation, and 10k/20k/30k checkpoints. Registered model-family differences:

- Diffusion Policy instead of ACT
- two observation frames
- horizon 48 (nearest value below ACT chunk 50 divisible by three-stage U-Net downsampling factor 8)
- DDPM, 100 train and 100 inference steps, epsilon prediction
- native MIN_MAX state/action normalization
- native Adam `1e-4`, betas `(0.95,0.999)`, weight decay `1e-6`
- cosine scheduler with 500-step warmup
- EMA enabled and used for deployment

Initial preflight failed only because `diffusers` was missing. After separately authorized installation of 0.39.0,
strict offline preflight passed. A separately authorized two-step GPU resource gate passed:

- 262,709,026 parameters
- loss 1.047/1.257; finite but too few steps for trend
- pre-clip grad norm 20.137/23.006
- peak PyTorch allocation 5.94 GiB
- no OOM/NaN/Inf

### Formal 30k training

Execution passed:

- 30,000 optimizer steps, 240,000 samples, about 10.42 effective epochs
- wall time 1:32:15
- 45.59 average samples/s
- peak PyTorch allocation 5.94 GiB
- first/last five train-loss mean: 0.1468 → 0.0218, down 85.15%
- first/last five pre-clip grad mean: 3.4758 → 0.3606
- no tracked 200-step window exceeded grad clip threshold 10
- live offline eval at 5k/10k/15k/20k/25k/30k:
  `0.0527/0.0481/0.0491/0.0557/0.0726/0.0761`
- live eval minimum was 10k, but EMA closed-loop quality improved through 30k; live offline loss is not a valid EMA
  deployment checkpoint selector
- live and EMA exports exist at 10k/20k/30k; checkpoint tree is approximately 15 GB

### Five-seed EMA checkpoint selection

All three checkpoints executed successfully but the preregistered development success gate failed:

| EMA checkpoint | Success | Mean sum | Mean max | Eval s/episode |
|---|---:|---:|---:|---:|
| 10k | 0/5 | 73.5272 | 0.58109 | 50.18 |
| 20k | 0/5 | 99.0972 | 0.60447 | 49.79 |
| 30k | 0/5 | 128.4161 | 0.85863 | 45.37 |

All success counts tied at zero, so the preregistered secondary metric selected and froze 30k EMA. The development
gate remains FAIL and was not retroactively changed by later held-out successes.

### Frozen 100-seed evaluation

Diffusion EMA 30k achieved 4/100 success on the same independent seeds used for ACT:

- exact 95% success interval: 1.10–9.93%
- successful seeds in that run: `20260860`, `20260901`, `20260924`, `20260929`
- mean/median sum reward: 86.8430/83.2745
- mean/median max reward: 0.65133/0.69303
- 3 zero-sum episodes
- 20 episodes with max reward ≥0.90; 5 with max reward ≥0.95
- 45.55 seconds/episode, about 39.0× deterministic ACT latency

Paired against deterministic ACT 30k:

- success 4% versus 6%; McNemar p=0.7266; bootstrap success difference `[-8pp,+3pp]`
- mean sum reward difference +14.489, bootstrap interval `[2.713,26.404]`
- mean max reward difference +0.13576, bootstrap interval `[0.06350,0.20880]`

Therefore Diffusion has stable dense-reward and peak-proximity improvements, but no stable sparse-success
improvement over deterministic ACT. The main remaining bottleneck is terminal threshold crossing/success retention,
plus severe 100-step DDPM latency.

### Demo and reproducibility boundary

The 100-seed evaluator rendered only episodes 0–9, none of which was a success. A standalone replay of environment
seed 20260860 failed because stochastic Diffusion sampling also depends on policy RNG history. A subsequent 31-episode
sequential rerun rendered all episodes and produced two new strict successes:

- episode 3, environment seed 20260833:
  `outputs/eval/phase1c_diffusion_pusht_ema_030000_replay_through_seed20260860/videos/pusht_0/eval_episode_3.mp4`
  SHA-256 `280bbc1a8c4bb0909ac8c2c443f2eeba2ff6ec363e1ddceea3509a072d748490`
- episode 8, environment seed 20260838:
  `outputs/eval/phase1c_diffusion_pusht_ema_030000_replay_through_seed20260860/videos/pusht_0/eval_episode_8.mp4`
  SHA-256 `31c5745eb0c2d29ca67e14912933fe94052b5fc6f8eb928d51cd53d0dbf6d258`

The target episode 30 did not reproduce and all first-ten video hashes differed from the original 100-seed run.
Current CUDA Diffusion trajectories are not bitwise stable across processes even with the same nominal seed sequence.
The two videos are valid frozen-policy success demos, but not exact reproductions of the original successful paths.

Key Diffusion records:

- `experiments/diffusion_policy/README.md`
- `results/phase1c_diffusion_pusht_baseline_30k.yaml`
- `results/phase1c_diffusion_pusht_baseline_30k_checkpoint_manifest.yaml`
- `results/phase1c_diffusion_pusht_checkpoint_selection_5ep.yaml`
- `results/phase1c_diffusion_pusht_ema_030000_100seed.yaml`
- `results/phase1c_diffusion_pusht_ema_030000_100seed_preregister.yaml`

## 6. Reproduction assets

Primary controlled entrypoint: `scripts/runs/pusht.sh`
Audit: `docs/pusht_reproducibility.md` and `results/pusht_run_asset_audit.yaml`

The script contains 28 strict-offline, overwrite-protected profiles covering ACT/Diffusion preflight, smoke/resource
gates, formal training, checkpoint accuracy, five-seed selection, ACT replanning diagnostics, paired 100-seed tests,
and Diffusion demo rendering.

Safe inspection:

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab
conda activate evla-core
scripts/runs/pusht.sh --list
EVLA_DRY_RUN=1 scripts/runs/pusht.sh diffusion-30k-train
```

Existing historical outputs cause profiles to refuse execution, preventing accidental overwrite. The raw logs on the
current host retain the exact `/usr/bin/time` command for 25 of 29 logs, but raw logs are not Git tracked. The new
profile script is the durable source for critical commands.

## 7. Current project-level conclusion

Completed:

- Phase 0 environment/core and LIBERO offline runtime gate
- Phase 1A PushT data contract
- Phase 1B ACT baseline and deterministic ablation
- Phase 1C Diffusion Policy baseline
- independent paired evaluation, success videos, structured manifests, and run-profile audit

PushT should not receive more same-data training by default. Evidence indicates:

1. deterministic ACT remains the best speed/success tradeoff at 6% and 1.17 s/episode;
2. Diffusion improves dense control/proximity but is 39× slower and only 4% successful;
3. VAE ACT is weakest at 2%;
4. terminal refinement, recovery-state coverage, and success retention are more important than merely extending
   training steps;
5. if returning to PushT, preregister either a denoising-step/latency inference ablation or a data-centric recovery
   experiment; do not tune on the held-out 100 seeds.

Recommended next project stage: Phase 2A task gate — select one short LIBERO task, pin demonstration/assets revision,
audit image/proprioception/action/termination contracts, and create an inert smoke config/checker before any training.
LIBERO data/model downloads and training still require separate authorization.

## 8. First checks for the next conversation

After reading this file completely, perform only these read-only checks first:

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab
git status --short --branch
git log -5 --oneline --decorate
git submodule status
```

Then read:

- `README.md`
- `docs/pusht_reproducibility.md`
- `experiments/act_baseline/README.md`
- `experiments/diffusion_policy/README.md`
- the structured result files listed above

Do not rerun completed experiments merely to verify them. Use tracked hashes and manifests unless the user explicitly
requests a rerun.

## 9. Copy-paste prompt for a new conversation

```text
请继续 Embodied VLA Lab 项目。

项目根目录：
/home/kxj/embodied_ws/embodied_vla_lab

请先完整读取最新总handoff：
/home/kxj/embodied_ws/embodied_vla_lab/docs/handoffs/2026-08-26_pusht_phase1c_complete_reproducibility_handoff.md

读取后只读核对：
- git status
- HEAD和最近五次commit
- 三个submodule pins

并按handoff中的“First checks”读取当前README、PushT复现审计、ACT/Diffusion README和结构化结果。
先用自己的话总结：当前已完成内容、ACT/Diffusion的100-seed核心结果、成功视频、结论边界、当前阻塞和
推荐下一阶段。不要重跑已完成实验。

后续我会继续粘贴宿主终端运行结果。每次收到输出后必须：
1. 首先判定PASS、FAIL或EXPECTED_EXCEPTION；
2. 分析loss/eval/grad/吞吐/显存/时间、reward/success/latency和结论边界；
3. 更新结构化结果、checkpoint manifest、对应README与可复现资产；
4. 最后只给下一条最小、安全宿主命令，或明确说明无需继续运行。

安装、dataset/model下载、新训练、GUI、W&B、Hub/upload、commit和push仍需分别明确授权。
同类只读检查可以自行执行，使用evla-core环境。不要使用multi-agent。

在我明确选择下一方向前，不要启动LIBERO下载/训练，也不要继续PushT调参。优先基于handoff建议讨论：
A. 进入LIBERO单任务数据/动作合同；
B. PushT Diffusion denoising-step/延迟消融；
C. PushT recovery/terminal-refinement数据方案。
```

## 10. Handoff creation boundary

Creating this document does not authorize or perform training, evaluation, download, installation, GUI, upload,
commit, or push. If it is to be frozen in Git, request separate commit authorization.

## 11. Phase 1D continuation update — WP1.1a metadata PASS

The next project direction was changed from immediate LIBERO work to Phase 1D PushT reproduction and diagnosis.
The first authorized network-only query resolved official reference repository `lerobot/diffusion_pusht` to full
revision `84a7c23178445c6bbf7e1a884ff497017910f653` and returned 8 files / 1,051,090,483 logical bytes. The model
is 1,050,862,408 bytes. No checkpoint payload, model, processor, environment, CUDA context, training, or rollout was
created.

The remote bundle contains `config.json` and `train_config.json` but lacks `policy_preprocessor.json` and
`policy_postprocessor.json`. Direct evaluation with pinned LeRobot 0.6.2 is therefore blocked pending a reviewed
legacy processor compatibility bundle. New records:

- `docs/phases/phase_1d_pusht_reproduction.md`
- `experiments/pusht_reproduction/README.md`
- `scripts/check_phase1d_pusht_reference_checkpoint.py`
- `results/phase1d_pusht_reference_checkpoint_metadata.yaml`
- `results/phase1d_pusht_reference_checkpoint_manifest.yaml`

Next gate: after separate download authorization, fetch only `README.md`, `config.json`, and `train_config.json` at
the frozen revision, hash them, and run the CPU/local-only static checker. Do not download the 1.05 GB weights in the
same gate.

The metadata-only download is now PASS: 3/3 files, 10,387 bytes, with SHA-256 identities frozen in the Phase 1D
result and manifest. The unauthenticated Hub warning affected rate limits only. Model weights remain absent. Next run
only the CPU/local static checker without `--require-weights`; exit 3 for missing legacy processor files is the
expected compatibility classification, not a policy-performance failure.

The checker result is now recorded: `EXPECTED_EXCEPTION`, exit 3, all 12 reference contract checks matched, and the
only compatibility risk is the missing processor pair. It used no network/CUDA/policy/environment. Raw log is 159
lines / 3,793 bytes with SHA-256 `e79aab51857bd4d2e70ad4c141dac5f29118b633edf1a4db3b3a5e6a8f224be2`.
WP1.1a is complete. Next perform a source-only audit of the old training-time normalization and serialization path;
do not download weights or synthesize processor statistics until that audit is complete.

The source audit found pinned LeRobot's intended `migrate_policy_normalization` utility. It extracts normalization
buffers from the legacy weights and writes a separate current-format model/processor bundle; therefore statistics
must not be guessed from current dataset metadata. The utility also performs a non-strict internal load and writes
new artifacts, so it is not yet authorized. Next perform only a read-only disk/cache preflight, then request separate
authorization for the 1.05 GB weight download and inspect safetensors keys/shapes before any migration.

The weight-download resource preflight is PASS: `/home` has 90 GB available at 43% use, the reference directory is
64 KB, and no dedicated reference Hub cache exists. Capacity does not block the 1,050,862,408-byte model. The next
step remains a separately authorized download into a new immutable source directory followed by hashes only; do not
load, migrate, initialize CUDA, or evaluate in that command.

The separately authorized source download is now PASS: four files, 1003 MB local `du`, approximately 4:59 transfer,
and model SHA-256 `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`. Metadata hashes match the prior bundle.
No load/migration/CUDA/environment/eval ran. Next use the local checker with `--require-weights` to inspect only
safetensors keys and shapes; missing processor JSON should keep the overall classification at expected exit 3.

The weight-header checker returned that expected exit 3. All 12 contracts match, 221 tensor headers are readable,
and archived SpatialSoftmax `[9,2]` exactly matches the pinned ResNet18 expectation. The prior `[144,2]` upstream
risk is not reproduced statically; missing processor JSON is now the only detected blocker. Log SHA-256 is
`d53ce96326a5270c9214c488ef365c64018eb806ca342a2839d9c87798ec45cf`. Next re-run the enhanced checker to enumerate
only legacy normalization keys/shapes; do not migrate or strictly load yet.

The normalization key/shape audit returned expected exit 3 and found exactly eight buffers matching the published
contract: image mean/std `[3,1,1]`, state min/max `[2]`, action-target min/max `[2]`, and action-output min/max `[2]`.
Log SHA-256 is `60c80e6967a7ed11670211a8ad4858dbe6bf90f21606c1a5c928f8d61eff4cf1`. Next read only these eight tensor values
and compare duplicated action statistics; do not construct or migrate the model.

The first Codex-side value command failed before checker startup because `evla-core` was not activated; the one-line
log is preserved. The retry is `EXPECTED_EXCEPTION` exit 3 and confirms complete finite stats: standard ImageNet RGB
mean/std, state min/max `[13.4564,32.9383]` / `[496.1462,510.9579]`, and identical target/output action min/max
`[12,25]` / `[511,511]`. Retry log SHA-256 is
`ac97751de86f78113f8ab4b6368cc4cb0b62d6ce7a9ee893e81b7974a657e739`. Processor migration is information-complete.
The next CPU policy-construction and full key/shape gate requires separate execution authorization; do not migrate or
run forward inference yet.

The authorized CPU schema gate is PASS: after excluding eight known normalization buffers, archived/current core
state dicts are both 213 tensors with zero missing, unexpected, or shape-mismatched keys. Construction was 0.904 s;
process wall 6.27 s; maximum RSS 1,948,656 KiB; swaps/sockets zero; CUDA stayed uninitialized. Log SHA-256 is
`9f90742691c19d9d1135ad4c84b91e0d9b1c03dcb1740055a7486f87dc8a7a56`. No values were loaded and no forward ran.
Next gate is strict core-weight loading on GPU without processors, forward, or environment and requires explicit GPU
execution authorization.

That authorized GPU gate is now PASS. After removing only the eight audited normalization buffers in memory,
`strict=True` loaded all 213 core tensors with zero missing/unexpected keys. All 262,709,026 parameters and buffers
were placed on `cuda:0` (RTX 5070 Ti Laptop GPU). CPU-to-CUDA transfer was 0.274 s; peak allocated/reserved VRAM was
1,057,328,640 / 1,059,061,760 bytes; total wall time 6.14 s; maximum RSS 3,099,392 KiB; swaps and sockets zero. The
61-line / 2,211-byte log SHA-256 is
`eda20d5da6d84f851374399cc699d0f327a3a1e536e38ad6bb8d0e36022e0e3b`. No processor, forward, environment,
loss, reward, success, gradient, or inference throughput was exercised. The next minimal gate is generation of a
separate current-format processor compatibility bundle from the eight verified legacy buffers; because it writes
approximately another checkpoint-sized artifact, it still requires explicit authorization. Do not modify the frozen
source directory or begin WP1.2 evaluation automatically.

The first authorized upstream migration attempt is FAIL at config serialization: JSON list values for tuple-typed
`crop_shape`, `down_dims`, and `optimizer_betas` remained lists, and draccus failed on integer `84`. The partial
directory has a zero-byte config and no model and must never be evaluated. Log SHA-256 is `ce667e62...d8f1`.

The controlled project-local v2 migration is PASS and independently verified. It strictly loaded 213 core tensors,
used the eight archived stats, wrote a nine-file 1,050,880,100-byte bundle, and preserved the frozen source. Every
migrated tensor is bit-exact to the source core; policy/preprocessor/postprocessor all reload offline. Generation and
verification hashes are `bec6550c...8d00` and `52666863...7d8`. Both runs were CPU-only with no forward/environment.
WP1.1 is complete. Next requires explicit authorization for WP1.2: one CUDA observation preprocess, one policy
action, postprocess and optionally one environment step, with no episode rollout.

WP1.2 is now PASS at development seed `20261001`. The v2 bundle loaded on CUDA, accepted a finite `[1,2]` action
`[94.3769,318.3038]` inside `[0,512]^2`, and PushT completed exactly one step. Reward was 0.0 and termination/
truncation were false, which is a valid first transition rather than an episode-performance result. Load was 1.860 s,
single cold action selection 1.613 s, peak allocated/reserved VRAM about 1104.1/1192.0 MiB, wall time 8.74 s, maximum
RSS 3,367,180 KiB, swap/network zero. Log SHA-256 is
`516a518659f5b7e9d7a1e6e1e765744576c18e36bd49d1d50a55d12884dd63b8`. No loss, gradient, full rollout, success
rate, return distribution, or steady-state throughput exists. Next preregister and separately authorize only a small
WP1.3 development episode set; do not touch historical/final seeds or begin large evaluation.

The inert WP1.3 profile is preregistered as `phase1d-reference-dev-5ep`: five episodes from evaluator seed `20261010`,
reference horizon 16 / execute 8 / 100 denoising steps, no recording, and new overwrite-protected paths. It remains
unauthorized and must not be run automatically.

WP1.3 is now PASS: evaluator seed `20261010`, 5 episodes, successes `[F,T,T,T,T]`, average sum reward 99.9626,
average max reward 0.999481, and 35.33 s/episode. Total eval time was 176.66 s and process wall 184.07 s; maximum
RSS 3,385,900 KiB, swap/network zero. Peak VRAM was not logged and must not be inferred from the earlier smoke.
The 80% Wilson 95% interval is about 37.6%--96.4%, so this is a strong development result but not a final estimate or
valid significance comparison against the frozen 100-seed baselines. Episode 0 max reward 0.9974 with success false
also confirms strict success-field accounting. Raw log SHA-256 is
`0106d2bb961fb5800a8daca6a279c5525e3adba21ac1c1f9fb787c051a0627bf`; all output hashes are frozen in
`results/phase1d_pusht_reference_dev_5ep.yaml`.

WP1.4 is preregistered before observation as exactly 100 never-used consecutive seeds `20261101..20261200`, with
the official reference contract unchanged. It requires separate final-evaluation authorization and must not run
automatically.

WP1.4 final is PASS and Phase 1D is complete. On `20261101..20261200`, the reference achieved 63/100 success
(Wilson 95% 53.2%--71.8%), mean sum/max reward 106.6265/0.947013, and 30.20 s/episode. Eval time was 3020.15 s;
wall 3026.68 s; maximum RSS 3,387,556 KiB; swaps/network zero; peak VRAM was not recorded. Against det-ACT 6/100,
VAE-ACT 2/100, and local Diffusion 4/100, success differences are 57/61/59 points with one-sided Fisher p about
`7.8e-19`, `4.8e-23`, and `9.4e-21`. These compare different fixed seed sets and assume independent Bernoulli
episodes. Ten failed episodes still reached max reward at least 0.99, so dense reward cannot replace strict success.

The result validates the evaluator/migration stack and rejects an inherent single-digit ceiling. It cannot isolate a
causal recipe factor because training budget, batch, crop, GroupNorm, horizon/action steps, normalization, and
checkpoint provenance differ jointly. Raw log SHA-256 is
`66872f08b3154cb54cb239e868b76c87a3a8231d9977a6320104e2c5a1ce7fd1`; structured evidence is
`results/phase1d_pusht_reference_final_100seed.yaml`. Next perform a read-only reproducibility audit, then request
separate commit authorization. Do not start LIBERO, training, upload, commit, or push automatically.

The final read-only reproducibility audit passed: four Phase 1D YAML records parse, six Phase 1D Python scripts parse,
`scripts/runs/pusht.sh` passes shell syntax and dry-run resolution, final log/JSON hashes match, HEAD remains
`121ba50`, and all three submodule pins remain unchanged. Local commit authorization was subsequently granted;
`env_backup_evla-core.yml`, checkpoint/output payloads, raw logs, and push remain outside the commit scope.
