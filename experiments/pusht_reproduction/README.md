# PushT reproduction, diagnosis, and recovery

Status: Phase 1D COMPLETE; preregistered official-reference final PASS at 63/100.

Phase 1D first verifies the official `lerobot/diffusion_pusht` checkpoint through the pinned local LeRobot and
PushT evaluator before considering any new training. Existing Phase 1B/1C results are frozen and will not be
overwritten or reinterpreted.

## Current reference gate

- repository: `lerobot/diffusion_pusht`
- frozen revision: `84a7c23178445c6bbf7e1a884ff497017910f653`
- remote inventory: 8 files / 1,051,090,483 logical bytes
- model: `model.safetensors`, 1,050,862,408 logical bytes
- remote policy/train configs: present
- new-style policy pre/postprocessor JSON: absent
- direct pinned LeRobot 0.6.2 evaluation: blocked pending an explicit processor compatibility bundle
- local metadata: 3 files / 10,387 bytes, fixed SHA-256 identities recorded

The first gate used network access only for metadata. The second downloaded `README.md`, `config.json`, and
`train_config.json` at the frozen revision; the unauthenticated warning affected only rate limits and all 3/3 files
completed. No model weights, policy, environment, CUDA context, training, or rollout were used. See
`results/phase1d_pusht_reference_checkpoint_metadata.yaml` and
`results/phase1d_pusht_reference_checkpoint_manifest.yaml`.

Next run the local static checker without `--require-weights`. The expected exit is 3 with
`BLOCKED_COMPATIBILITY_REVIEW_REQUIRED`, because the official bundle lacks processor JSON. Do not interpret that
expected compatibility result as policy-performance failure or download the 1.05 GB weights in the same gate.

The checker has now produced exactly that expected result: exit 3, all 12 contract checks matching, no network/CUDA/
policy/environment side effects, and raw-log SHA-256
`e79aab51857bd4d2e70ad4c141dac5f29118b633edf1a4db3b3a5e6a8f224be2`. WP1.1a is complete. The current task is
source-level review of the legacy normalization/serialization path; do not manufacture processor statistics or
download weights until that review determines what the official checkpoint actually expects.

Pinned LeRobot provides `lerobot.processor.migrate_policy_normalization`, which extracts legacy normalization buffers
from `model.safetensors` and writes a separate current-format checkpoint plus processors. This establishes the intended
compatibility path, but the utility mutates a copy and uses a non-strict internal load. It will not be run directly.
The next gate is a disk/cache preflight followed by separately authorized weight download and read-only safetensors
key/shape inspection; migration requires its own later authorization and stricter acceptance criteria.

The disk/cache preflight is PASS: `/home` has 90 GB free at 43% use, the current reference directory is 64 KB, and
no dedicated `models--lerobot--diffusion_pusht` Hub cache exists. Capacity does not block the 1,050,862,408-byte
model. The next download remains a separate authorization and will write a new immutable `source` directory; it will
not load, migrate, or evaluate the policy.

The separately authorized source download is PASS: four files completed at the frozen revision, the directory uses
1003 MB, and the 1,050,862,408-byte model SHA-256 is
`995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`. All metadata hashes match their prior copies.
Approximate transfer time was 4:59 at a displayed 3.01 MB/s. No weights were loaded and no migration, CUDA,
environment, or evaluation ran. Next run only the CPU safetensors metadata checker with `--require-weights`.

That checker returned the expected exit 3 solely because processor JSON is absent. All 12 contracts still match;
the archive contains 221 tensors, and its SpatialSoftmax `pos_grid` `[9,2]` exactly matches the pinned ResNet18
expectation `[9,2]`. The reported upstream `[144,2]` mismatch is therefore not reproduced in this static local
contract. The checker is now extended to enumerate legacy normalization tensor keys/shapes on the next read-only run.

The normalization tensor audit returned the same expected exit 3 and found exactly eight buffers: image mean/std
`[3,1,1]`, state min/max `[2]`, action-target min/max `[2]`, and action-output min/max `[2]`. Their categories and
shapes exactly implement the published VISUAL MEAN_STD plus STATE/ACTION MIN_MAX contract. Values have not yet been
read. The next checker revision reads only these eight tiny tensors to verify finite values, ImageNet-stat intent, and
equality of target/output action statistics.

The first automated value-audit attempt failed before checker startup because the Codex shell had not activated
`evla-core`; its one-line log is preserved. The retry explicitly activated the environment and returned the expected
exit 3. Image mean/std are standard ImageNet RGB values, state min/max are finite, and action min/max are `[12,25]`
and `[511,511]`; target-normalization and output-unnormalization action values match exactly. The legacy checkpoint
therefore contains a complete processor-stat contract. The next gate requires CPU policy construction and full
non-normalization key/shape comparison, so it needs separate execution authorization.

The authorized CPU schema gate is PASS. After excluding only the eight known normalization buffers, all 213 archived
core tensors match all 213 tensors constructed by pinned LeRobot: zero missing, unexpected, or shape-mismatched keys.
Construction took 0.904 seconds; the full process took 6.27 seconds with 1,948,656 KiB maximum RSS, zero swap, zero
socket traffic, and no CUDA initialization. No tensor values were loaded into the model and no forward ran. The next
gate is a strict core-weight value load on the deployment GPU, still without processors, forward, or environment.

The authorized GPU gate is also PASS: `strict=True` loaded all 213 core tensors with zero missing/unexpected keys,
and all 262,709,026 parameters/buffers reside on `cuda:0`. CUDA transfer took 0.274 seconds, peak allocation was
1,057,328,640 bytes, total wall time was 6.14 seconds, and maximum RSS was 3,099,392 KiB. The immutable log SHA-256
is `eda20d5da6d84f851374399cc699d0f327a3a1e536e38ad6bb8d0e36022e0e3b`. No forward, processor, environment,
loss, reward, success, gradient, or inference-throughput measurement ran. Next is a separately authorized write of
an out-of-tree processor compatibility bundle; the official source directory remains immutable.

The upstream migration attempt failed at config serialization after writing only partial processor assets; that
invalid directory is retained. A controlled v2 migration fixed the list/tuple reconstruction path without modifying
the submodule. Its nine-file 1003 MB bundle reloads offline, both processors reload, and every one of 213 core tensor
values exactly equals the frozen source. This completes WP1.1 compatibility but provides no action/reward/success
evidence. Next is one separately authorized CUDA observation-to-action smoke, not an episode rollout.

That WP1.2 smoke is PASS at seed `20261001`: observation and action shapes match, the finite action
`[94.3769,318.3038]` is inside `[0,512]^2`, and PushT accepted exactly one step. Action selection took 1.613 s and
peak allocated VRAM was about 1104.1 MiB. Reward 0.0 on the first transition is not an episode-level failure; no
success/return distribution was measured. The next separately authorized gate is a small development-only episode
set, with final/historical seeds excluded.

The five-episode development run succeeded 4/5 with average sum/max reward 99.9626/0.999481 and 35.33 s/episode.
Its 80% Wilson 95% interval is roughly 37.6%--96.4%, so it validates the compatibility/evaluation path but is not a
final performance estimate. The never-used final range `20261101..20261200` is now frozen before observation and
requires separate authorization.

The final never-used range `20261101..20261200` produced 63/100 strict successes, mean sum/max reward
106.6265/0.947013, and 30.20 s/episode. Wilson 95% is 53.2%--71.8%. This decisively validates the compatibility and
evaluator path relative to all frozen single-digit baselines, while the combined recipe differences prevent
single-factor causal attribution. See `results/phase1d_pusht_reference_final_100seed.yaml`.
