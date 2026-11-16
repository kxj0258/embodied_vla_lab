# Phase 2A — LIBERO single-task data/action contract

Updated: 2026-08-27  
Current gate: Phase 2B 100k training PASS; independent trained-policy environment rollout not yet run.

## Scope and authorization boundary

Phase 2A establishes one language-conditioned manipulation contract before any ACT-on-LIBERO training. The selected
minimal task is `LIBERO_GOAL/turn_on_the_stove`, task-order 0 index 7. This is a one-predicate articulated-object task,
so it minimizes task sequencing complexity while exercising language, dual-camera vision, proprioception, 7D control,
and an explicit simulator success predicate.

The current assets are inert. They do not authorize demonstration download, LIBERO top-level initialization,
environment construction/reset/step/render, GPU forward, training, formal evaluation, GUI, W&B, upload, commit, or push.

## Frozen static identity

- LIBERO submodule: `8f1084e3132a39270c3cc3e7e98f04940126398a391`;
- suite/task: `libero_goal/turn_on_the_stove`;
- language: `Turn on the stove`;
- BDDL goal: `(And (Turnon flat_stove_1))`;
- BDDL SHA-256: `cbd2a0514d53226c5d09e17af40b22935d4cb6ff8fdaff0d55f555de1adecc13`;
- pruned init states: NumPy array `[50,79]`, `float64`, SHA-256
  `ed121db31d749647d70741c4c4ef0bd8e0876663986578795968e962d9b91340`.

All five pinned registries have matching BDDL/pruned-init counts: spatial 10, object 10, goal 10, LIBERO-90 90,
and LIBERO-10 10.

## Candidate runtime contract

- Panda, robosuite 1.4.0 `OSC_POSE`;
- normalized 7D action: XYZ delta, axis-angle rotation delta, gripper command, each accepted in `[-1,1]`;
- controller scaling: translation approximately `[-0.05,0.05]`, rotation `[-0.5,0.5]`;
- 20 Hz control; five initial zero-action physics-settle steps; maximum 600 policy steps;
- two 128x128 RGB streams: `agentview_image` and `robot0_eye_in_hand_image`;
- low-dimensional state: 2D gripper qpos plus 7D joint position;
- sparse reward 1 on BDDL success; step termination is overridden by the same `_check_success()` result.

These values are source-derived and remain a candidate until an authorized runtime gate confirms actual observation
keys/shapes, action spec, frequency, termination, and rendering in the exact Phase 2A wrapper.

## Demonstration data contract

The single official file was downloaded from `yifengzhu-hf/LIBERO-datasets` at frozen revision
`f13aa24a3da8c43c7225569f28c562979fa0e35a`. It is 447,509,922 bytes with SHA-256
`387fc10747696b80dea6ed8d7f2beaa162bf92ae11750b241073cbd33aac73d5`.

It contains 50 demonstrations / 4,460 frames. Episode lengths range from 75 to 119 (mean 89.2). Both camera streams
are 128x128x3 uint8; gripper/joint/EE states are 2D/7D/6D float64; simulator and robot state widths are 79 and 9.
Actions are finite 7D float64 values within `[-1,1]`. Every recorded episode has reward sum 1 and done sum 1.

The first timed download was truncated, and a redirected resume produced a readable file with a duplicated tail.
That invalid 455,632,290-byte artifact is preserved as `.invalid-resume-duplicate` and is never a dataset input.
The canonical path contains only the fresh byte-exact download.

The simulator asset cache exists at `/home/kxj/.cache/libero/assets` and occupies about 409 MiB.

`LIBERO_CONFIG_PATH` is unset and `/home/kxj/.libero/config.yaml` is absent. The pinned LIBERO top-level module prompts
and writes that file when imported without an existing configuration. Static tooling therefore parses pinned files
directly and deliberately avoids importing LIBERO. A later runtime gate must use a project-local, explicit config path
and must not silently mutate the user home directory.

## Evidence and metric boundary

The audit executes no model loss, policy forward, environment replay/rollout, gradient calculation, throughput
benchmark, GPU allocation, or statistical test. Recorded reward/done fields prove dataset labels only, not current
runtime success. The audit establishes byte/schema/action-range integrity but not replay compatibility or generalization.
The final static/data checker took 1.32 seconds with maximum RSS 672,620 KiB; this is file-audit resource use, not
policy or environment throughput. CUDA was not initialized and VRAM was not measured.

## Demo 0 isolated EGL replay

The fixed `demo_0` initial simulator state loaded exactly. All 80 recorded 7D actions were accepted, both camera
streams remained valid, success first occurred at step 70, final success was true, and replay reward sum was 11.
This establishes an interface/task-semantic replay PASS for one recorded demonstration.

The preregistered deterministic next-state threshold failed: maximum L2 was 0.507810 versus the required 0.01.
Drift began after the first action; qpos max/median L2 were 0.049906/0.027594 and qvel max/median were
0.507635/0.032192. The threshold must not be relaxed after observing the result. Current MuJoCo 3.8.1 / robosuite
1.4.0 therefore does not reproduce archived simulator states exactly, even though it reproduces task success.
The diagnostic replay took 11.70 seconds (13.32 process wall) and maximum RSS was 2,264,316 KiB. No policy, loss,
gradient, training, or formal evaluation was involved; N=1 is not a success-rate estimate.

Canonical assets:

- `configs/env/libero_goal_turn_on_stove_phase2a.yaml`;
- `scripts/check_phase2a_libero_static_contract.py`;
- `results/phase2a_libero_single_task_static_contract.yaml`;
- `results/phase2a_libero_single_task_contract_manifest.yaml`.

## Next gates

1. Run the inert static/data checker in `evla-core` and freeze its output.
2. Make an explicit compatibility decision: require legacy-state determinism, or accept task-semantic replay while
   preserving the failed deterministic subgate.
3. Only after that decision define the dataset split/preprocessing contract. GPU policy forward, training, and
   evaluation remain later independent gates.

## Accepted compatibility decision and dataset contract

The user accepted task-semantic replay while permanently preserving deterministic next-state replay as FAIL.
A seeded 40/10 demonstration split contains 3,571/889 frames. Training-only mean/std statistics are frozen for
9D joint+gripper state and 7D action. ACT samples use two CHW float RGB views, one 9D state, a 20x7 action chunk,
and a 20-element padding mask. The read-only checker passed in 0.34 seconds / 54,592 KiB RSS.

The first two-step GPU resource attempt failed before model construction because CUDA is unavailable in the managed
container. No loss, gradient, throughput, or VRAM result exists from that attempt. Run the same gate on the host.

The host rerun PASSed: 34,185,031 parameters, losses 0.617429 to 0.456378, finite pre-clip gradient norms
112.81/86.73 with clip 10, 2.994 optimizer steps/s, 706,334,208-byte peak allocated VRAM, and 2,379,912 KiB maximum
RSS. This proves resource fit and finite optimization only; it is not an offline validation or success result.

The separately identified 200-step single-batch overfit run is FAIL: loss fell 0.617429 to 0.175013 (71.65%) but
did not meet the preregistered greater-than-75% reduction. Its early `classification: PASS` print was a reporting-order
bug contradicted by exit 1; the script now prints metrics and the computed classification before asserting. Grad,
throughput, and VRAM tail metrics were not emitted by that run and must not be inferred from the two-step gate.

The new 400-step run with the unchanged threshold PASSed: loss 0.617429 to 0.135600 (78.04% reduction), minimum
0.122852; pre-clip grad norm fell 112.81 to 14.98, throughput was 49.22 steps/s, peak allocated VRAM 706,334,208
bytes, process wall 13.07 seconds, and maximum RSS 2,379,076 KiB. It remains a memorization test only.

Full-split smoke v1 completed 20 finite optimizer steps but FAILed at checkpoint serialization because pinned
`ACTConfig` has no `to_dict()`. No checkpoint exists; the preserved output contains only resolved config and metrics.
The two logged losses are different random batches and are not a convergence comparison. Serialization now uses
`dataclasses.asdict`; rerun under a new v2 experiment ID.

The Codex-managed v2 attempt is an EXPECTED_EXCEPTION: CUDA peak-memory counter reset failed before the first
batch or optimizer step. It generated no loss, gradient, throughput, VRAM, checkpoint, evaluation, reward, or
success evidence, and therefore does not test the serialization fix. Its log and resolved config are preserved;
the host rerun must use a new v3 experiment ID.

Host v3 PASSed 20 full-split optimizer steps and checkpoint serialization. The 410,450,149-byte checkpoint reloads
on CPU with step 20, 178 model tensors, optimizer state, ACT config dict, and frozen normalization statistics.
Peak allocated/reserved VRAM was 709,839,360/834,666,496 bytes; process wall was 6.89 seconds and maximum RSS was
2,454,016 KiB. No offline validation, rollout, success, or reward ran, so it is smoke-only. The next gate adds the
previously configured but missing deterministic full validation-set loss before any medium-scale training.

Validation smoke v4 PASSed over all 889 frozen validation frames with sample-weighted loss 0.742955 at step 20.
The checkpoint is byte-identical to v3, confirming evaluation is non-mutating. This is offline action loss only,
not success or reward. The entrypoint now records step-0 validation and resolves environment overrides into the
saved config, enabling a 1,000-step medium-scale trend gate with validation every 250 steps.

The user selected a 100,000-step formal budget. The entrypoint now saves model/optimizer plus Python, NumPy,
PyTorch, CUDA, and DataLoader-generator state. Periodic checkpoints are every 4,460 steps (ten complete 446-step
epochs), making them exact-resume boundaries; the non-boundary final checkpoint is archival and is not accepted for
exact continuation. Full offline validation runs every 1,000 steps. Environment success/reward rollout remains an
independent checkpoint evaluation so simulator failure cannot terminate training.

## Phase 2B formal 100k result

`phase2b_act_libero_turn_on_stove_deterministic_100k_seed20260826` completed all 100,000 optimizer steps and is
classified PASS. Sparse logged train loss fell from 0.917988 at step 1 to 0.070784 at step 100,000; its minimum
logged value was 0.048785 at step 81,000 and the last ten logged points averaged 0.065300. Pre-clip gradient norm
was finite throughout the sparse log, peaked at 89.894 on step 1, and the last ten points averaged 3.356. Of 1,001
logged gradient points, 65 exceeded the clip boundary of 10; this is not an exact clipped-step count because
intermediate steps were not logged.

All 101 scheduled validations completed over all 889 frozen validation frames. Loss fell from 1.016288 at step 0
to the run minimum 0.426634 at step 1,000, then remained on an approximately 0.43--0.47 plateau and ended at
0.440030. Continued train-loss reduction therefore did not yield monotonic validation improvement. This complete
offline action-loss curve is not environment success, reward, or proof that step 1,000 is the best control policy.

The process reported 3,348.79 seconds training elapsed, 3,354.04 seconds wall time, about 29.86 optimizer steps/s,
709,839,360/834,666,496 bytes peak allocated/reserved VRAM, and 2,558,808 KiB maximum RSS. It produced 22 periodic
exact-resume checkpoints and one non-boundary archival step-100k checkpoint. All 23 are 410,470,545 bytes, have no
`.pt.incomplete` sibling, contain every required state field, and reload on CPU; hashes are frozen in
`results/phase2b_act_libero_checkpoint_manifest.yaml`.

This is one training seed and one fixed offline split. No trained-policy LIBERO environment rollout ran, so success,
reward, rollout throughput, failure taxonomy, environment checkpoint ranking, and multi-seed uncertainty remain
unmeasured. The next gate requires a new experiment ID and explicit environment-evaluation authorization.

## Phase 2B step-100k development rollout

The independent experiment `phase2b_act_libero_ckpt100000_dev5_seed20260827` is PASS. All five fixed development
initial states reached strict LIBERO success in 78--83 environment steps; every episode had sum/max reward 1.0.
This validates the complete trained-policy path: 180-degree camera correction, frozen joint-then-gripper 9D state,
checkpoint statistics, normalized action prediction, action denormalization/clipping, five-action execution queue,
relative OSC control and the BDDL success predicate.

The five episodes used 403 total environment steps in 17.456 s, or about 23.09 environment steps/s and 3.491 s per
episode. There were 82 policy forward calls. Timed selection totaled 0.989 s including the first CUDA cold start;
the four warm episodes averaged about 7.81 ms per forward. Peak allocated/reserved VRAM was
162,552,832/216,006,656 bytes; process wall was 24.06 s and maximum RSS was 4,020,488 KiB. Training loss, offline
validation, gradients and optimizer throughput were not remeasured by this evaluation.

This is a strong development result, not a final 100% success estimate: N=5 gives a Wilson 95% interval of roughly
56.6%--100%, the five initial states were used for development, and only one checkpoint was evaluated. The next
paired gate applies the identical five-state protocol to the earliest periodic checkpoint at step 4,460. A final
held-out run must use a subsequently frozen checkpoint and must not be used for further selection.

The paired step-4,460 development evaluation also executed successfully and is classified PASS, while its policy
result was 4/5 strict successes. Init states 0--3 succeeded in 74, 72, 115 and 163 steps; init state 4 reached the
600-step horizon with zero reward. Mean sum/max reward was 0.8. In contrast, step 100,000 succeeded on the same
state 4 in 78 steps and was 5/5 overall. Thus the paired development evidence favors step 100,000 by one success,
0.2 mean reward and lower all-episode mean steps (80.6 versus 204.8).

This difference is not statistically established: there is only one discordant pair. To avoid evaluating all 23
checkpoints and overfitting selection to five states, the remaining development candidate is preregistered as the
midpoint step 49,060 checkpoint. After that run, selection must freeze using success first, then reward, then fewer
steps; the subsequent held-out final set cannot be used to revise selection.

The third and final development candidate, step 49,060, achieved 5/5 strict successes and mean reward 1.0 in
82/82/81/83/79 steps (407 total). Step 100,000 also had 5/5 and reward 1.0 but used 403 total steps, so the frozen
lexicographic rule selects step 100,000. The four-step margin is only a development tie-break and is not evidence of
statistical superiority. No further checkpoints may enter selection.

There are only 50 pinned task init states. Since indices 0--4 were used for development, a never-used 50-state final
is impossible. The uncontaminated final protocol is therefore frozen as all remaining 45 states, indices 5--49,
with evaluator seeds 20260901--20260945. This held-out result will estimate performance over the remaining finite
init-state set and cannot revise the selected checkpoint.

## Phase 2B held-out final

The preregistered final `phase2b_act_libero_ckpt100000_final_heldout45_seed20260901` is PASS. The frozen step-100k
checkpoint succeeded on all 45 init states excluded from development, indices 5--49. Every episode had strict BDDL
success and sum/max reward 1.0. Completion steps totaled 3,584: mean 79.64, median 79, range 74--106 and P95 87.

Pure rollout took 81.612 s (1.814 s/episode; 43.92 environment steps/s). The evaluator made 733 policy forward
calls; timed action selection was 6.028 s total, about 8.22 ms/forward including CUDA cold start and 7.77 ms warm.
Peak allocated/reserved VRAM was 162,552,832/216,006,656 bytes. Process wall was 87.83 s and maximum RSS was
4,178,080 KiB. Evaluation-only fields—training loss, offline validation, gradients and optimizer throughput—were
not remeasured.

This is an exact enumeration of the complete 45-state finite held-out set, so 45/45 is exact for that set. A Wilson
95% interval of approximately 92.13%--100% is relevant only when treating episodes as Bernoulli samples from a
broader population. The five development states also had 5/5 for the selected checkpoint, giving descriptive 50/50
coverage of all pinned states, but only 45/45 is uncontaminated final evidence. The result does not establish
multi-task, language, camera-perturbation, simulator-version or multi-training-seed generalization. Phase 2B is complete.
