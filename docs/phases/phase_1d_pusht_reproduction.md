# Phase 1D — PushT reproduction, diagnosis, and recovery

Updated: 2026-08-26
Current gate: Phase 1D COMPLETE; preregistered final evaluation PASS at 63/100.

## Scope and authorization boundary

Phase 1D tests whether the current PushT evaluation stack can reproduce a strong public Diffusion Policy before
new training is considered. Existing Phase 1B/1C records are frozen. Every new run must use a new experiment ID,
output directory, raw log, structured result, and manifest.

This document and `scripts/check_phase1d_pusht_reference_checkpoint.py` are inert assets. They do not authorize a
checkpoint download, policy construction, CUDA use, environment reset/step, rollout, training, upload, commit, or
push. The checker reads only a user-supplied local directory and never contacts the Hub.

## WP0 — frozen local baselines

Repository boundary at the start of Phase 1D:

- branch/HEAD: `main` at `121ba50`;
- dataset: `lerobot/pusht` at `7628202a2180972f291ba1bc6723834921e72c19`;
- split: 185 train episodes / 23,043 frames and 21 validation episodes / 2,607 frames;
- environment: PushT-v0, 96x96 RGB, 2D agent state, 2D absolute-position action, 10 Hz, 300 steps;
- pinned LeRobot: `8b256a6c0d4769c3cc3e7e98f04940126398a391`, package version 0.6.2.

Frozen independent results on environment seeds `20260830..20260929`:

| Frozen model | Success | Mean sum reward | Mean max reward | Eval seconds/episode |
|---|---:|---:|---:|---:|
| deterministic ACT 30k | 6/100 | 72.3539 | 0.51557 | 1.168 |
| VAE ACT 25k | 2/100 | 67.3235 | 0.41226 | 1.394 |
| Phase 1C Diffusion 30k EMA | 4/100 | 86.8430 | 0.65133 | 45.55 |

The Phase 1C result is a same-data lightweight/ACT-compatible model-family comparison, not an official recipe
reproduction. It used horizon 48, execute 10, no crop, no GroupNorm, batch 8, 30k steps, and 100-step DDPM. Its
dense-reward improvements are observed facts; recipe mismatch as the cause of low success remains a hypothesis.

## WP1.1 reference identity and published contract

Candidate reference repository: `lerobot/diffusion_pusht`. Its model card says it was trained with LeRobot commit
`3c0a209`, batch 64, 200k steps, 25k evaluation/save cadence, seed 100000, and evaluated at checkpoint 175k over
500 episodes. The authorized metadata query resolved and froze repository revision
`84a7c23178445c6bbf7e1a884ff497017910f653`; `main` must not be used as the experiment identity.

The query returned 8 files / 1,051,090,483 logical bytes. `model.safetensors` accounts for 1,050,862,408 bytes.
The separately authorized metadata-only download then fetched `README.md`, `config.json`, and `train_config.json`:
3 files / 10,387 bytes. Their SHA-256 identities are recorded in the structured result and manifest. The public-Hub
unauthenticated warning affected rate limits only; 3/3 files completed. No model weights were downloaded, and no
policy, environment, CUDA context, or rollout was created. Structured evidence is in
`results/phase1d_pusht_reference_checkpoint_metadata.yaml` and the manifest in
`results/phase1d_pusht_reference_checkpoint_manifest.yaml`.

Published `config.json` contract as observed on 2026-08-26:

- policy type Diffusion; image `[3,96,96]`, state `[2]`, action `[2]`;
- two observation steps, horizon 16, execute 8, drop last 7 frames;
- random 84x84 crop in training and center crop in evaluation;
- ResNet18 without ImageNet pretrained backbone weights, GroupNorm enabled;
- visual MEAN_STD and state/action MIN_MAX normalization;
- DDPM, 100 train timesteps, inference steps unset (therefore defaults to train timesteps), epsilon prediction;
- Adam `1e-4`, betas `(0.95,0.999)`, weight decay `1e-6`, cosine schedule and 500-step warmup.

The archived `train_config.json` says `dataset.use_imagenet_stats=true` even though the policy normalization mapping
still says visual MEAN_STD. This means the exact serialized normalization statistics are essential; it is unsafe to
substitute the current Phase 1C dataset statistics by assumption.

## Local contract versus reference contract

| Field | Phase 1C local baseline | Official/reference checkpoint | Compatibility meaning |
|---|---|---|---|
| policy / features | Diffusion; image 3x96x96, state 2, action 2 | same | shape/key contract matches |
| action semantics | absolute target `[x,y]` | PushT dataset/action 2D; expected absolute | verify through processor and finite action smoke |
| observation steps | 2 | 2 | matches |
| horizon / execute | 48 / 10 | 16 / 8 | checkpoint values must win on first evaluation |
| crop | disabled | random 84x84 train; center 84x84 eval | model-internal config, not environment crop |
| GroupNorm | off | on | architecture-affecting; cannot override |
| normalization | visual dataset MEAN_STD; state/action MIN_MAX | visual ImageNet-stat training intent; state/action MIN_MAX | serialized processor/stat source must be recovered exactly |
| scheduler | DDPM 100/100, epsilon | DDPM 100/default 100, epsilon | effectively matches |
| backbone weights | none | none | matches |
| training budget | batch 8, 30k | batch 64, 200k | performance comparison, not load compatibility |
| LeRobot code | 0.6.2 / `8b256a6c` | trained at `3c0a209` | high-risk architecture/serialization gap |

## What current `lerobot-eval` restores

The pinned `EvalPipelineConfig` reads `config.json` through `PreTrainedConfig.from_pretrained` whenever
`--policy.path` is supplied. Explicit `--policy.*` CLI/YAML values are applied as overrides, so the first reference
evaluation must override only deployment device/AMP and must not override horizon, action steps, crop, GroupNorm,
normalization, or features.

`make_policy` then uses the loaded config and checkpoint weights. The environment supplies its runtime feature map;
because the checkpoint already has non-empty input features, it does not replace those shapes. The project must still
pass explicit PushT 96x96 environment fields: pinned `PushtEnv` declares 384x384 by default while gym-pusht emits
96x96, and its `gym_kwargs` does not forward observation dimensions. This declaration mismatch can fail environment
feature checks even when runtime pixels are correct.

After model creation, current `lerobot-eval` calls `make_pre_post_processors` with the same pretrained path. It
expects `policy_preprocessor.json` and `policy_postprocessor.json` and only overrides the device step and optional
observation rename map. Environment-specific PushT processors are identity pipelines. Thus environment options do
not legitimately replace checkpoint normalization or crop; crop is implemented inside the Diffusion RGB encoder.

## Known blockers to direct evaluation

The current Hub file list contains `config.json`, `train_config.json`, `model.safetensors`, and evaluation/demo
artifacts, but no new-style policy pre/postprocessor JSON files. A current upstream issue reports that direct
LeRobot 0.6.1 evaluation fails with `FileNotFoundError` for those files. The pinned 0.6.2 factory follows the same
unconditional load path, so direct `lerobot-eval --policy.path=lerobot/diffusion_pusht` is expected to fail before
rollout. This is a compatibility blocker, not evidence of a bad environment or policy.

A second upstream issue reports that LeRobot 0.4.4 constructed a `[144,2]` SpatialSoftmax grid while this checkpoint
archived `[9,2]`, causing weight-load and forward failures. The pinned 0.6.2 source keeps the checkpoint's non-empty
96x96 input feature declaration and constructs the grid from the post-crop backbone feature map; for an 84x84
ResNet18 input the static expectation is also `[9,2]`. The reported failure may therefore have been fixed by later
feature/config handling, but the exact archived shape and a real strict load still need verification. `strict=False`
is not an acceptable silent fix: it does not prove forward semantics and would change the reference contract without
a registered compatibility decision.

An older, closed upstream issue also documents a Gymnasium 1.0 success-accounting regression. The local Phase 1B/1C
pipeline has already recorded strict successes under gym-pusht 0.1.6, so it is not known to suffer that exact zero-only
failure. Nevertheless WP1.2 must explicitly compare reward, termination, and success fields rather than treating a
single aggregate success number as sufficient validation.

## Reference checkpoint evaluation implementation plan

1. **WP1.1a metadata pin (network, no weights):** remote identity query and the three-file local metadata bundle PASS
   at full revision `84a7c231...f653`; sizes and SHA-256 are frozen. Next run the static checker without
   `--require-weights`. Expected outcome is the published contract plus
   `BLOCKED_COMPATIBILITY_REVIEW_REQUIRED` because processors are absent.
2. **WP1.1b weight identity (about 1.05 GB):** after separate checkpoint-download authorization and disk check,
   download `model.safetensors` at the same frozen revision. Hash it and run the checker with `--require-weights`.
   This opens safetensors metadata on CPU only and reports the archived SpatialSoftmax grid shape without loading
   tensors or constructing a model.
3. **Compatibility decision:** choose and document one reproducible adapter path. Preferred direction is an
   out-of-tree compatibility bundle that preserves the official `config.json` and weights byte-for-byte while adding
   explicitly generated processor files from the checkpoint's training dataset/stat contract. Any SpatialSoftmax
   adapter must have a code-level explanation and fixed hashes. Do not edit the Hub snapshot or pinned submodule.
4. **WP1.2 load/action smoke:** only after review and separate execution authorization, load on CUDA, restore the
   approved processors, reset one PushT environment, preprocess one observation, predict/postprocess one action, and
   take one step. Check finite/range/shape, block movement, reward, termination, memory, and latency. No episode rollout.
5. **WP1.3 development episodes:** preregister new development seeds and run only a small number of episodes. Preserve
   reference horizon 16, execute 8, 100 denoising steps, center-crop evaluation, and serialized normalization.
6. **WP1.4 final sets:** freeze a never-used 100-seed final set before observation, while retaining
   `20260830..20260929` only as the historical paired set. Checkpoint selection may not use either final set.

WP1 passes only if load/processor/action semantics/environment/success accounting are credible and the reference
checkpoint performs clearly above the frozen single-digit baselines. If the reference remains at single-digit
success after compatibility and smoke gates pass, stop before WP2 and debug the version/environment/evaluator stack.

### WP1.1a static checker result

The local checker completed with exit 3 and `BLOCKED_COMPATIBILITY_REVIEW_REQUIRED`, classified as
`EXPECTED_EXCEPTION`. All 12 published-contract checks matched: policy type, image/state/action features,
observation/action horizons, crop, GroupNorm, normalization mapping, scheduler, timesteps, and prediction target.
The only reported risk was the known absence of policy pre/postprocessor JSON. It used no network, CUDA, policy, or
environment. Raw log: 159 lines / 3,793 bytes, SHA-256
`e79aab51857bd4d2e70ad4c141dac5f29118b633edf1a4db3b3a5e6a8f224be2`.

This closes metadata identity and contract validation, but not processor compatibility. Before weight download, the
next review must establish where the old LeRobot training commit stored image/state/action normalization statistics
and whether the current processor migration utilities can reproduce that exact path without modifying official
config or weights.

### Legacy processor migration source audit

The training commit abbreviated as `3c0a209` is not present in the shallow local LeRobot submodule, so its exact old
implementation cannot be recovered locally without a separate source-history network action. The pinned 0.6.2 source
does, however, include the intended migration utility:
`lerobot.processor.migrate_policy_normalization`.

That utility loads the legacy `model.safetensors`, extracts built-in normalization buffers for image/state/action,
removes those normalization keys from a copy of the state dict, creates current pre/postprocessor pipelines with the
extracted statistics, reconstructs the policy, and writes a separate migrated checkpoint directory. It can also push
to Hub, but Phase 1D must not use that option. This confirms that the exact processor statistics are expected to come
from the legacy weights, not be guessed from `train_config.json` or substituted from the current Phase 1C dataset.

The migration is a mutating, full-weight operation and internally uses `strict=False` while reporting missing and
unexpected keys. It must therefore not be run automatically after download. First inspect the weight archive and
normalization keys read-only; then preregister an out-of-tree migration directory, require zero problematic model-key
differences, and compare migrated normalization values with the published training contract. Official downloaded
files remain immutable and separately hashed.

### WP1.1b weight download resource gate

The read-only host preflight is PASS: `/home` reports 164 GB total, 66 GB used, 90 GB available, and 43% use. The
current reference directory occupies 64 KB and the dedicated Hugging Face cache for `lerobot/diffusion_pusht` is
absent. The 1,050,862,408-byte remote model therefore has ample capacity headroom. This is not download authorization
and provides no evidence about transfer integrity, model load, GPU memory, inference, reward, or success.

The separately authorized download subsequently completed all four requested source files. The immutable source
directory uses 1003 MB; `model.safetensors` is 1,050,862,408 bytes with SHA-256
`995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`. The metadata hashes match the earlier bundle.
Transfer took approximately 4:59 with displayed download/reconstruction throughput of 3.01/5.04 MB/s. The
unauthenticated warning is an expected rate-limit notice because 4/4 files completed. No model load, tensor-schema
audit, migration, CUDA, environment, or evaluation occurred.

The first weight-header checker completed as `EXPECTED_EXCEPTION` exit 3 because processor JSON remains absent. It
read 221 tensor headers and found `diffusion.rgb_encoder.pool.pos_grid` shape `[9,2]`, exactly matching the pinned
ResNet18 static expectation `[9,2]`. Thus the upstream `[144,2]` construction mismatch is not reproduced for this
frozen config/source contract. This removes that static architecture blocker but does not substitute for strict model
loading. Raw log: 171 lines / 4,008 bytes, SHA-256
`d53ce96326a5270c9214c488ef365c64018eb806ca342a2839d9c87798ec45cf`.

The checker has been extended to enumerate normalization tensor keys and shapes without loading tensor payloads into
a policy. That is the next gate before any migration or strict load.

The normalization tensor audit then found exactly eight legacy buffers: image mean/std `[3,1,1]`, state min/max
`[2]`, action-target min/max `[2]`, and action-output min/max `[2]`. This exactly matches the published normalization
mode contract and the migration utility's supported key patterns. It remained `EXPECTED_EXCEPTION` exit 3 solely
because processors are absent. Raw log: 226 lines / 5,075 bytes, SHA-256
`60c80e6967a7ed11670211a8ad4858dbe6bf90f21606c1a5c928f8d61eff4cf1`.

Only shapes and keys were observed in that run. The next local checker reads the eight tiny values and compares the
duplicated action target/output statistics; it still does not instantiate a policy or migrate files.

The automated value-audit first produced a clean harness FAIL because the Codex process had not activated
`evla-core`; `python` was not found and the checker never started. That one-line log is retained. The retry explicitly
activated `evla-core` and returned the expected compatibility exit 3.

The eight values are finite. Image mean/std are the standard ImageNet RGB values (subject only to float32 display),
confirming `train_config.dataset.use_imagenet_stats=true`. State min/max are
`[13.4564238,32.9382935]` / `[496.1461792,510.9578857]`. Action min/max are `[12,25]` / `[511,511]`, and target
normalization equals output unnormalization exactly. Processor migration is therefore information-complete and must
use these archived values. Retry log: 288 lines / 6,168 bytes, SHA-256
`ac97751de86f78113f8ab4b6368cc4cb0b62d6ce7a9ee893e81b7974a657e739`.

This still does not establish full model compatibility. The next gate constructs the pinned policy on CPU and
compares all non-normalization parameter names/shapes without forward inference. Because that allocates the model and
crosses beyond ordinary metadata reading, it requires separate authorization.

The authorized CPU-only schema gate is PASS. The official archive contains 221 tensors: eight known normalization
buffers and 213 core model tensors. Pinned LeRobot constructs exactly 213 core tensors, with 213 shared keys and zero
missing, unexpected, or shape-mismatched entries. Policy construction took 0.904 seconds; total wall time was 6.27
seconds, maximum RSS 1,948,656 KiB, swaps zero, sockets zero, and CUDA remained uninitialized. The CPU fallback
warnings were expected for this isolated gate. Log SHA-256:
`9f90742691c19d9d1135ad4c84b91e0d9b1c03dcb1740055a7486f87dc8a7a56`.

This proves exact architecture schema compatibility but did not load archived tensor values. The next separable gate
removes only the eight known normalization keys in memory and strictly loads all 213 core tensors on the deployment
GPU, recording missing/unexpected keys and peak allocation without forward inference.

The separately authorized strict GPU gate is **PASS**. All 213 core tensor values loaded with `strict=True`, with
zero missing and zero unexpected keys. All 262,709,026 parameters and buffers moved to the RTX 5070 Ti Laptop GPU.
The CUDA transfer took 0.274 seconds; peak allocated/reserved memory was 1,057,328,640 / 1,059,061,760 bytes. Total
wall time was 6.14 seconds, maximum RSS was 3,099,392 KiB, and swap/network counts were zero. Raw log: 61 lines /
2,211 bytes, SHA-256 `eda20d5da6d84f851374399cc699d0f327a3a1e536e38ad6bb8d0e36022e0e3b`.

This closes core architecture and value-load compatibility only. No processor, forward pass, environment, loss,
reward, success, gradient, inference throughput, or statistical performance was measured. The next gate is creation
of a separate current-format compatibility bundle from the eight audited normalization buffers. That operation
writes about another checkpoint-sized artifact and still requires explicit authorization; the frozen source bundle
must remain unchanged.

The first authorized run of the upstream migration utility is **FAIL**, preserved as evidence. It wrote processor
files but failed before model save because raw JSON lists for `crop_shape`, `down_dims`, and `optimizer_betas` were
not converted back to tuple-typed config fields; `draccus` failed while encoding integer `84`. Its partial directory
has a zero-byte config and no model and is explicitly invalid.

The project-local controlled v2 migration is **PASS**. It uses `PreTrainedConfig.from_pretrained`, strict core loading,
the eight source stats, and an atomic new directory. The 1003 MB / 1,050,880,100-byte bundle contains nine files.
Independent verification proves all 213 migrated tensors are bit-exact to the source core tensors and reloads the
policy plus both processors offline. Generation/verification log hashes are `bec6550c...8d00` and
`52666863...7d8`. No forward or environment ran. WP1.1 is complete; WP1.2 requires a separately authorized CUDA
single-observation/single-action smoke before any episode evaluation.

WP1.2 is **PASS** at development seed `20261001`. The verified bundle produced one finite float32 action
`[94.3769,318.3038]` with shape `[1,2]`, inside PushT's `[0,512]^2` bounds, and the environment accepted exactly one
step. The first-transition reward was 0.0 with neither termination nor truncation. Policy/processor load took 1.860 s
and one cold 100-step Diffusion action selection took 1.613 s; peak allocated/reserved VRAM was approximately
1104.1/1192.0 MiB. Total wall time was 8.74 s and maximum RSS 3,367,180 KiB, with no swap or network. Raw log
SHA-256: `516a518659f5b7e9d7a1e6e1e765744576c18e36bd49d1d50a55d12884dd63b8`.

This establishes the complete interface through one environment transition, not policy quality. Reward 0.0 at the
first step cannot be classified as episode failure, and no success rate, return/max-reward distribution, steady-state
throughput, loss, gradient, or statistical conclusion exists. WP1.3 must use a small preregistered development-only
seed set and requires separate evaluation authorization.

WP1.3 development gate is preregistered as profile `phase1d-reference-dev-5ep`: five sequential closed-loop episodes
from evaluator seed `20261010`, with reference horizon 16, execute 8, 100 denoising steps, no recording, and a new
output/log path. It is inert until separately authorized.

WP1.3 is **PASS**. Five development episodes from evaluator seed `20261010` produced successes
`[false,true,true,true,true]`, average sum reward 99.9626, average max reward 0.999481, and 35.33 seconds/episode.
Episode 0 was not successful despite max reward 0.9974, so strict success accounting cannot be replaced by a dense
reward threshold. The observed 80% has a Wilson 95% interval of approximately 37.6%--96.4%; it is a strong
development signal, not a final rate or a significance comparison with the frozen 100-seed baselines. Log SHA-256
is `0106d2bb961fb5800a8daca6a279c5525e3adba21ac1c1f9fb787c051a0627bf`; all five videos and `eval_info.json` are
hashed in `results/phase1d_pusht_reference_dev_5ep.yaml`.

Before observing any final result, WP1.4 is preregistered as the never-used consecutive range
`20261101..20261200`, 100 episodes, unchanged reference horizon 16 / execute 8 / 100 denoising steps. It requires
separate final-evaluation authorization and must be run exactly once into new overwrite-protected paths.

WP1.4 final is **PASS** on that preregistered range. The reference produced 63/100 successes (Wilson 95%
53.2%--71.8%), average sum reward 106.6265, average max reward 0.947013, and 30.20 seconds/episode. This is
57/61/59 percentage points above deterministic ACT, VAE ACT, and local 30k Diffusion; one-sided Fisher exact
p-values are approximately `7.8e-19`, `4.8e-23`, and `9.4e-21`. Ten failed episodes nevertheless reached max reward
at least 0.99, reinforcing the strict success boundary.

This closes Phase 1D: the evaluator/migration stack can reproduce a strong public PushT policy and is not inherently
capped at single-digit success. It does not isolate why local models underperformed because training budget, batch,
crop/GroupNorm, horizon/action window, normalization, and checkpoint provenance changed together. The final log
SHA-256 is `66872f08b3154cb54cb239e868b76c87a3a8231d9977a6320104e2c5a1ce7fd1`; structured evidence is in
`results/phase1d_pusht_reference_final_100seed.yaml`.

## Checker contract and expected statuses

```bash
python scripts/check_phase1d_pusht_reference_checkpoint.py \
  --checkpoint-dir /absolute/path/to/frozen/reference/metadata

python scripts/check_phase1d_pusht_reference_checkpoint.py \
  --checkpoint-dir /absolute/path/to/frozen/reference/checkpoint \
  --require-weights
```

The checker performs no network, CUDA, model construction, or environment operation. Exit `0` means the local static
bundle has no detected blocker. Exit `2` means required local files or the published contract do not match. Exit `3`
means the known legacy processor or architecture risks require an explicit compatibility decision. For the current
official repository, exit `3` is the expected WP1.1 result, not a failed policy-performance result.

## Evidence boundary

Observed facts above come from the frozen local repository/source, the public checkpoint file list/config/model card,
and upstream issue reports. Issue reports establish credible compatibility risks, not proof that every platform or
future revision fails identically. The checkpoint bytes, exact schema, strict core load, and CUDA placement are now
verified, and the full forward/environment/final evaluation path is validated on the preregistered 100-seed set.
Generalization beyond this PushT environment and causal attribution among recipe differences remain unknown.

## Public references

- Official checkpoint and model card: <https://huggingface.co/lerobot/diffusion_pusht>
- Published policy config: <https://huggingface.co/lerobot/diffusion_pusht/raw/main/config.json>
- Published training config: <https://huggingface.co/lerobot/diffusion_pusht/raw/main/train_config.json>
- Missing processor pipeline report: <https://github.com/huggingface/lerobot/issues/4047>
- SpatialSoftmax compatibility report: <https://github.com/huggingface/lerobot/issues/3802>
- Historical Gymnasium success-accounting report: <https://github.com/huggingface/lerobot/issues/470>
