# Embodied VLA Lab — Phase 3B complete total-project resume

Date: 2026-08-27  
Project root: `/home/kxj/embodied_ws/embodied_vla_lab`  
Purpose: latest conversation-resume entry after completion of Phase 3B. This document supplements, and does not
replace, the cumulative historical fact source
`docs/handoffs/2026-08-26_phase1d_complete_total_project_handoff.md`.

## 1. Required reading order for a new conversation

Read these files completely, from start to finish, before taking project action:

1. `docs/handoffs/2026-08-26_phase1d_complete_total_project_handoff.md` — cumulative history and immutable facts.
2. `docs/handoffs/2026-08-27_phase3a_openvla_single_image_complete_resume.md` — Phase 3A continuation history.
3. `docs/handoffs/2026-08-27_phase3b_complete_total_project_resume.md` — this latest state, priorities and boundary.

Then perform only the read-only checks requested in the new-conversation prompt. Reading handoffs does not authorize
training, evaluation, download, installation, GUI, upload, commit, push or deletion.

## 2. Repository and protection boundary

- Git HEAD: `f0c0368fd07b51928bb1b96be2c7e0fc45818182`.
- LeRobot pin: `8b256a6c0d4769c3cc3e7e98f04940126398a391`.
- LIBERO pin: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- OpenVLA pin: `c8f03f48af692657d3060c19588038c7220e9af9`.
- The worktree is intentionally dirty/untracked with project assets. Do not clean, reset, delete or overwrite it.
- Preserve `env_backup_evla-core.yml` and all existing outputs, raw logs, manifests and failure directories.
- Frozen Phase 1/2 finals, Phase 3A/3B results and all historical failures are immutable.
- Every new run needs a new experiment ID, output directory, raw log, structured result and manifest entry.
- Existence guards must either use explicit `if` branches or reliable errexit. A prior `test ! -e` without `set -e`
  allowed an accidental v2 rerun and raw-log overwrite; that protocol collision is permanently documented.
- Short local reads/searches/hashes, CPU loads, static checks and project-file writes are authorized. Formal training,
  long GPU/environment runs, large downloads and environment changes require one complete host command for the user.
- Do not use GUI, W&B, Hub/upload, commit, push or deletion without new explicit authorization.
- The user explicitly requested no multi-agent work.

## 3. Project mission and completed phase map

The project is a simulation-first embodied-AI engineering and research portfolio built around LeRobot, PushT,
LIBERO and OpenVLA on one 12 GB RTX 5070 Ti Laptop GPU. The long route is imitation learning → language-conditioned
manipulation → VLA inference/adaptation → viewpoint robustness → small world model → Feagine continuum-arm transfer.

Completed:

- Phase 0: isolated runtimes, CUDA/BF16, MuJoCo EGL, pinned upstream repositories and reproducibility conventions.
- Phase 1A--1D: PushT dataset audit, ACT/VAE ACT/Diffusion training, checkpointing, 100-seed evaluation and videos.
- Phase 2A--2B: LIBERO task/data contract, ACT 100k training, full offline validation, checkpoint selection and final.
- Phase 3A: official OpenVLA NF4 inference, dependency-isolated environment bridge and complete base-policy final.
- Phase 3B: rank-8 QLoRA resource/training pipeline, paired offline evaluation, development rollout and immutable final.

Not started: Phase 4 ego-view/viewpoint robustness, Phase 5 small world model and Phase 6 Feagine transfer.

## 4. Frozen Phase 1 and Phase 2 conclusions

### PushT

- Deterministic ACT: 6/100 independent-seed success.
- VAE ACT: 2/100.
- Diffusion EMA: 4/100.
- Official PushT reference: 63/100.
- Diffusion has better paired dense reward/proximity but no better sparse success and is about 39× slower than
  deterministic ACT with the tested 100-step DDPM inference.
- Do not add more same-data PushT training without a new preregistered hypothesis.

### ACT on LIBERO `turn_on_the_stove`

- Frozen 40/10 demonstration split: 3,571/889 frames.
- ACT 100k training PASS; 101 full validations. Minimum loss 0.426634 at step 1,000; final 0.440030.
- Development checkpoint selection: step 100k selected after early/middle/final candidates.
- Held-out init states 5--49: 45/45 strict success, mean reward 1.0, mean/median steps 79.64/79, range 74--106.
- This is single-task/fixed-runtime evidence, not language, viewpoint, simulator or real-world generalization.

## 5. Phase 3A frozen OpenVLA base result

- Official model: `openvla/openvla-7b-finetuned-libero-goal`, revision
  `fa5ae1e7509348889295bba8e08621d8b55e9baf`.
- Snapshot: 16 files, 15,085,050,797 logical bytes, hash-manifest verified.
- NF4/BF16 real-image inference PASS. Single cold action about 1.024 s; peak allocated/reserved about 4.71/4.84 GiB.
- Implemented `evla-core` environment coordinator + `evla-openvla` persistent model worker with hashed PNG/JSON IPC,
  schema checks, timeout, action adaptation, image rotation/crop and deterministic cleanup.
- Development init0--4: 5/5 success.
- Held-out init5--49: 45/45 strict success; mean/median steps 91.87/76, P95 101.4.
- State 12 required 593/600 steps, demonstrating a severe base-policy efficiency tail despite perfect success.
- Median/P95 inference about 0.409/0.433 s in the frozen Phase 3A final.
- The frozen base remains the recommended reliable demo/deployment policy after Phase 3B.

## 6. Phase 3B QLoRA implementation and result

### Resource and training implementation

- Installed only upstream-pinned `peft==0.11.1` with `--no-deps`; Torch/Transformers/Accelerate remained frozen.
- NF4/BF16, rank 8, alpha 8, dropout 0, `target_modules=all-linear`, Gaussian initialization.
- 27,707,072 trainable parameters, 0.7048% of 3,931,114,624 parameters with adapters.
- One-sample forward/backward PASS: loss 17.8423, finite gradient norm 539.24, peak allocated/reserved
  6,621,465,600/6,851,395,584 bytes. This was an untrained resource baseline, not validation or environment evidence.
- One clipped AdamW update PASS: max norm 1.0, 218,256,208-byte optimizer state, finite nonzero parameter update.
- Dependency isolation was preserved by exporting HDF5 samples to hash-frozen PNG/JSON in `evla-core`; no HDF5
  dependency was added to `evla-openvla`.

### Canonical 20-step smoke

- Canonical experiment: `phase3b_openvla_qlora_rank8_batch1_train20_val10_v3_seed20260827`.
- Train first/last loss: 22.7779/12.3476; first/last-10 mean: 14.9103/10.3167.
- Val10 loss: 16.9151→8.8018; token accuracy: 4/70→9/70.
- Twenty finite clipped optimizer updates completed.
- Five checkpoint files hash-verified; optimizer/RNG CPU-load PASS at step 20 with 852 state entries.
- Peak allocated/reserved VRAM: 6,755,235,328/7,128,219,648 bytes.
- Canonical checkpoint manifest: `results/phase3b_openvla_qlora_train_smoke_v3_checkpoint_manifest.yaml`.

### Full paired offline validation

Identical 889 frames, 10 episodes and 6,223 action tokens:

| Metric | Frozen base | Step-20 adapter | Change |
|---|---:|---:|---:|
| mean loss | 16.9597 | 8.9271 | -47.36% |
| action-token accuracy | 9.64% | 19.97% | +10.33 percentage points |
| continuous-action L1 | 0.3077 | 0.2073 | -32.63% |

- All three metrics improved on 10/10 validation episodes.
- Loss improved on 883/889 frames; L1 improved on 732/889.
- Offline adaptation classification: PASS.
- Canonical paired result: `results/phase3b_openvla_step20_vs_base_fullval889_paired.yaml`.
- These metrics are not environment reward/success.

### Development environment

- Single init0: success at step 102; frozen base succeeded at step 85. N=1 did not establish efficiency change.
- Dev init0--4: adapter 5/5 success with steps [102,77,73,71,72], total 395/mean 79.0.
- Frozen base: [85,87,78,86,85], total 421/mean 84.2.
- Adapter aggregate steps were 6.18% lower; faster on 4/5 states, slower by 17 steps on init0.
- Development evidence justified freezing step-20 for exactly one held-out final, without further tuning.

### Immutable held-out final

- Complete init5--49 enumeration, no post-final tuning permitted.
- Execution PASS; policy success 43/45 = 95.56%.
- Wilson 95% interval: 85.17%--98.77% when treated as Bernoulli observations.
- Failures: init12 and init44, both at 600 steps/reward 0.
- Frozen base succeeded those states in 593 and 68 steps respectively.
- Adapter capped mean/median/P95: 95.24/70/123.4; frozen base: 91.87/76/101.4.
- Total capped steps: adapter 4,286; base 4,134.
- On the 43 common-success states, adapter was faster/slower/tied on 32/10/1 with mean/median delta -9/-6 steps.
- Interpretation: typical successful control improved, but two catastrophic tails reduced reliability from 45/45 to
  43/45 and worsened capped mean/P95.
- Phase 3B conclusion: engineering feasibility PASS; offline adaptation PASS; environment non-regression FAIL.
- Step-20 adapter is a research artifact, not the reliable demo/deployment candidate.
- Canonical final: `results/phase3b_openvla_qlora_step20_final_heldout45_v1.yaml`.
- Canonical comparison: `results/phase3b_openvla_qlora_final_comparison.yaml`.

## 7. Important preserved failure chain

Do not delete or rewrite these failures; they are useful engineering evidence:

- QLoRA resource v1: forward/backward completed, writer failed because model output shadowed output path.
- Optimizer gate v1: checked the first LoRA parameter, which validly had zero first-step update; insufficient sentinel.
- Train smoke v1: `main()` was defined but never invoked; exit 0 with zero work.
- Train smoke v2 first attempt: CUDA OOM after initial val10 before step 1.
- Accidental v2 rerun: old ID/log reused because shell guard lacked errexit; compute passed but protocol classification
  is FAIL and its checkpoint is quarantined.
- Base fullval v1: direct model incorrectly used a PEFT wrapper access path; zero validation frames.
- Adapter dev5 v1: external GPU contention left 23.94 MiB free; adapter load OOM before worker readiness/episode 1.
- All fixes used new IDs. Later commands use explicit guards and, for adapter environment runs, an 8,000 MiB
  free-VRAM preflight.

## 8. Existing demonstrable assets

- PushT ACT and Diffusion successful videos already exist with checkpoint/video manifests.
- ACT-on-LIBERO has a complete 45/45 final, trajectories, training/validation history and checkpoint manifest; it is
  ready to package into a polished video but no unified final presentation video has yet been produced.
- OpenVLA base has single-image, single-step, single-episode, dev5 and 45/45 held-out trajectories; it is the strongest
  reliable VLA demo candidate and is ready for headless video packaging.
- QLoRA has a complete engineering/offline/environment comparison and two final failure trajectories. It is best
  presented as a research analysis demo: offline and typical-efficiency gains versus tail-robustness regression.
- Do not claim that a polished ACT/OpenVLA LIBERO demo video already exists unless the relevant rendering/composition
  asset is actually created and hash-manifested.

## 9. Resume/portfolio-ready claims

Safe claims:

- Built a reproducible LeRobot/LIBERO/OpenVLA embodied-learning lab covering data contracts, ACT/Diffusion training,
  full offline validation, checkpoint selection and closed-loop MuJoCo evaluation.
- Trained ACT for 100k steps on one LIBERO task and obtained 45/45 strict success on a frozen held-out finite set.
- Ran OpenVLA-7B in NF4 on a 12 GB laptop GPU and implemented a dependency-isolated hashed IPC control bridge;
  frozen base also achieved 45/45 held-out strict success on the same task/runtime.
- Implemented rank-8 QLoRA over 0.7048% trainable parameters; paired 889-frame validation improved loss 47.36%, token
  accuracy 10.33 points and L1 32.63%, while final closed-loop success regressed from 45/45 to 43/45.
- Diagnosed the mismatch between teacher-forced offline objectives and closed-loop tail robustness instead of
  selecting a model solely by loss.

Unsafe claims:

- Do not claim general LIBERO 100%, language generalization, viewpoint robustness, real-world transfer or QLoRA
  superiority. All strong results are one task, one fixed simulator/runtime and fixed finite init-state sets.

## 10. Recommended next work, in priority order

### P0 — MVP presentation and reproducibility closure

Do this before new research training:

1. Synchronize the README top status and project summary to Phase 3B complete.
2. Build a headless, reproducible OpenVLA base success demo from a frozen trajectory/run; base is the reliable policy.
3. Build an ACT-on-LIBERO representative success video.
4. Build a research comparison video/figure: base success versus QLoRA typical speedup and init44 tail failure.
5. Add instruction, model, step, reward/success and timing overlays; create video SHA-256 manifests.
6. Create one architecture diagram and one quantitative results figure/table.
7. Add a read-only result-verification entrypoint and audit that referenced artifacts/hashes exist.
8. Produce a concise project report/portfolio page and final resume bullets.

No GUI is authorized. Use headless/offline rendering and new immutable output IDs. If a long environment rerun is
unnecessary because frozen images/trajectories suffice, prefer composing from existing assets.

### P1 — Phase 4 ego-view/viewpoint robustness

- Define a new development/final protocol that does not reuse Phase 3B held-out outcomes for tuning.
- Fix task/model/checkpoint first; vary only camera/viewpoint/crop/occlusion/blur/lighting/extrinsics.
- Compare agent-view, eye-in-hand and controlled perturbations.
- Report success, steps, failure tails and latency, not only image/action loss.

### P2 — Multi-task and language robustness

- Add 3--5 LIBERO tasks and paraphrased instructions with new development/held-out splits.
- Test layout and visual perturbations.
- Avoid describing one-task results as broad VLA generalization.

### P3 — Small action-conditioned latent world model

- Frozen/small image encoder; predict latent future from latent + action.
- Compare copy-frame, no-action and action-conditioned baselines.
- Measure one/multi-step error and simple candidate-action ranking; do not train a large video generator.

### P4 — Feagine simulation transfer

- Start with language/image → high-level target → action adapter → PCC IK/controller → MuJoCo.
- First task: safe low-speed move-to-target with workspace/action bounds and emergency stop.
- Add obstacle avoidance/contact only after simulation gates pass; real hardware remains outside current evidence.

## 11. Immediate next-action boundary

Phase 3B is complete. Do not run more Phase 3B training, rerun its held-out final, replace its selected checkpoint or
tune from init12/init44. The next recommended implementation is P0: audit and create reproducible headless demo and
portfolio assets using the frozen base policy and existing results. Before any rendering work, inspect existing
trajectory/image/video assets and define new output IDs/manifests. No host command is currently pending.

## 12. Operating protocol for future pasted host output

For every pasted terminal result:

1. Classify PASS, FAIL or EXPECTED_EXCEPTION first.
2. Analyze every available loss, offline validation, success, reward, gradient, throughput, VRAM, RSS, timing,
   checkpoint-integrity and statistical-boundary field.
3. Never equate offline loss/token accuracy with environment success/reward.
4. Update structured results, checkpoint/model manifest, README, roadmap, phase docs, cumulative handoff and
   reproducibility assets.
5. End with exactly one minimal safe independent host command, or explicitly stop at the authorization boundary.

## 13. P0 MVP implementation update (latest; 2026-08-27)

P0 is `PARTIAL_PASS`. Canonical offline assets are in `outputs/presentation/p0_mvp_v2_20260827` and include:

- OpenVLA frozen-base held-out init5 strict-success replay (83 steps);
- init40 base-versus-QLoRA typical speedup (221 versus 70 steps, both success);
- init44 base success at 68 versus QLoRA failure at the frozen 600-step cap;
- paired fullval889 metrics, held-out reliability and system architecture figures in PNG/SVG;
- SHA-256/ffprobe manifest plus a read-only verifier.

The build used frozen PNG/JSON only: environment steps and model inferences were both zero. The 10 fps encoding is
presentation timing, not measured control/inference throughput. V1 is preserved as noncanonical visual-QA history;
v2 corrected image scaling/title overlap under a new output ID.

README, roadmap, `docs/phases/p0_mvp_presentation.md`, `docs/portfolio.md`, `docs/resume.md`, the cumulative handoff
and this resume are updated. ACT final evidence has no saved frames, so a policy video cannot honestly be composed
without a recapture. The new-ID one-episode headless capture config/script are statically ready but unexecuted. It
reuses frozen checkpoint/init5 only for presentation and cannot revise the prior 45/45 result. Completing that one
host capture, incorporating its hash into the presentation inventory and rerunning verification are the only
remaining P0 media steps.

## 14. P0 COMPLETE after ACT presentation capture (latest; 2026-08-27)

The one pending ACT capture is PASS: frozen step-100k on held-out init5 succeeded in 83 steps with reward 1.0.
Seventeen policy forwards used 0.5748 s timed inference; peak allocated/reserved VRAM was
162,552,832/216,006,656 bytes, host wall 21.44 s and maximum RSS 3,826,808 KiB. Checkpoint runtime hash/strict load
PASSed; no loss/offline-validation/grad/optimizer/checkpoint-creation metric applies.

The original capture is valid evidence but has a clipped title. It remains preserved; a new-ID layout-only offline
recomposition is the canonical video and performed zero environment steps/model inference. P0 is now COMPLETE.
Canonical manifest: `results/p0_mvp_v2_manifest.yaml`. Next priority is P1, but no viewpoint experiment, environment
run, training, download, commit or push is authorized merely by this completion.

## 15. P0-HD 640px upgrade started (latest; 2026-08-28)

The user approved a presentation-quality upgrade using genuine 640x640 LIBERO rendering. This is not post-hoc AI
super-resolution. High-resolution display capture is separated from frozen policy input: ACT deterministically
rotates/downsamples to 128x128; OpenVLA will preserve its frozen crop/model preprocessing. All runs are new-ID
presentation recaptures and cannot revise frozen Phase 2/3 evidence.

The ACT init5 640px coordinator/config are implemented and static PASS with zero environment/CUDA execution. It
produces a 1920x1080, 20 fps, CRF-16 H.264 video under a new output path and records checkpoint hash, trajectory,
success, reward, inference, VRAM and timing. Formal host execution is pending. OpenVLA base/QLoRA HD capture remains
to be implemented after this render/policy-decoupling gate proves stable. No training, evaluation-final rerun,
download, installation, commit or push occurred.

## 16. P0-HD ACT gate PASS; canonical dual-environment design (latest; 2026-08-28)

ACT 640 capture PASSed at 82 steps/reward 1.0 with clear 1920x1080 output. Seventeen forwards used 1.0195 s;
VRAM allocated/reserved was 162,552,832/216,006,656 bytes, host wall 24.45 s, maximum RSS 3,913,920 KiB and no
swap/network. Loss/offline-validation/grad/optimizer/checkpoint creation are not applicable.

Because this first gate downsampled a 640 renderer into the 128 policy input, it is a noncanonical quality gate, not
an exact-input replay. Canonical captures now use a native 128 policy environment plus a state-synchronized
render-only 640 environment. The generic OpenVLA coordinator and five base/QLoRA init5/40/44 configs are static
PASS. Base init5 is the next formal host gate. Frozen Phase 3 results remain immutable.

## 17. P0-HD OpenVLA v1 infrastructure FAIL; single-EGL v2 static PASS (latest; 2026-08-28)

The base-init5 v1 host window closed during dual-EGL environment initialization. This is `FAIL`, specifically an
infrastructure failure before policy execution: the raw log contains only robosuite warnings; the worker never
started, the model never loaded, environment steps are zero, and no success/reward, inference, VRAM, RSS, elapsed
footer, video or success result exists. The only partial output is the resolved config. Afterwards `nvidia-smi`
could not communicate with the driver. Preserve the v1 output/log and
`results/p0_hd_openvla_base_init05_v1_failure.yaml`; never retry that ID or interpret it as policy evidence.

The dual-environment design is superseded. V2 keeps one native-128 LIBERO environment and sends its unchanged native
observation through the frozen OpenVLA IPC preprocessing. A second direct 640 camera render comes from the same
MuJoCo simulation and same EGL offscreen context, so there is no second EGL environment or state copy. Script syntax,
schema, single-environment construction and fresh paths are static PASS under
`configs/eval/p0_hd_openvla_base_init05_v2.yaml`. Do not execute it until the NVIDIA driver is restored and
`nvidia-smi` passes.

## 18. NVIDIA driver recovery PASS; OpenVLA held for GPU contention (latest; 2026-08-28)

After reboot, `nvidia-smi` PASSed: driver 580.173.02, reported CUDA 13.0, GPU temperature 52 C and no driver
communication error. The recovery check performed no OpenVLA load/inference, environment step, success/reward
measurement or checkpoint validation. A separate `/home/kxj/miniforge3/envs/unitree_rl/bin/python` process (PID
35905) occupied 3381 MiB; total GPU use was 3990/12227 MiB at 64% utilization. Therefore v2 formal capture remains
deferred to avoid OOM risk and contaminated latency. Do not terminate that unrelated process automatically.
Structured record: `results/p0_hd_gpu_recovery_gate_20260828.yaml`.

## 19. P0-HD GPU resource gate PASS (latest; 2026-08-28)

At 00:41:05 the unrelated `unitree_rl` compute process was gone. NVIDIA-SMI reported driver 580.173.02, GPU
620/12227 MiB used (11607 MiB derived free), 12% utilization and graphics processes only. Classify the resource
preflight PASS and allow exactly the new-ID base-init5 single-EGL v2 presentation capture next. This check itself
has no model load, environment success/reward, inference/resource timing or checkpoint-integrity evidence. Record:
`results/p0_hd_gpu_resource_gate_20260828_004105.yaml`.

## 20. P0-HD OpenVLA frozen-base init5 v2 PASS (latest; 2026-08-28)

The replacement single-sim/single-EGL design execution and policy success are PASS: frozen base init5 completed in
83 steps/reward 1.0 with 83 predictions. Model load was 4.619 s; inference total/mean/median/P95 was
28.437/0.3426/0.3369/0.3434 s (2.919 predictions/s). Peak allocated/reserved VRAM was
5,061,286,912/5,196,742,656 bytes; internal elapsed 57.46 s, host wall 61.67 s, max RSS 5,805,596 KiB, zero swap and
zero network messages. The worker matched frozen revision `fa5ae1e...baf`, loaded four shards and shut down cleanly.

The presentation/clean videos are H.264 yuv420p at 20 fps, 83 frames and 1920x1080/1080x1080 respectively; their
hashes plus the 83-row trajectory are in `results/p0_hd_openvla_base_init05_v2.yaml`. Visual QA PASSed genuine 640
detail and clean layout. Training/offline/grad/optimizer metrics are non-applicable, and this recapture cannot revise
the Phase 3A 45/45 final. Next is the new-ID frozen-base init40 half of the planned typical-speedup HD pair.

## 21. P0-HD OpenVLA frozen-base init40 v2 PASS (latest; 2026-08-28)

Frozen base init40 execution and strict success PASSed at the expected 221 steps/reward 1.0 with 221 predictions.
Load was 5.509 s; inference total/mean/median/P95 was 74.359/0.3365/0.3366/0.3434 s (2.972 predictions/s). Peak
allocated/reserved VRAM was 5,061,286,912/5,196,742,656 bytes; host wall 132.18 s, max RSS 5,775,076 KiB and zero
swap/network. Revision, four shards, clean worker shutdown and post-run GPU cleanup PASSed.

The 221-frame 1920x1080 presentation and 1080x1080 clean videos pass H.264/yuv420p/20fps, hash and visual checks.
This exactly establishes the base half of the frozen init40 221-versus-70-step presentation comparison; it adds no
new selection evidence. The fresh QLoRA init40 v2 contract changes only the frozen worker/policy and is next. Never
use it to tune the adapter or revise Phase 3B 43/45.

## 22. P0-HD QLoRA init40 v2 PASS; typical-speedup pair complete (latest; 2026-08-28)

Frozen adapter init40 execution/policy success PASSed at 70 steps/reward 1.0, reproducing the known outcome. Against
base 221, this is 151 fewer steps (-68.33%; 3.16x base/adapter step ratio), but adapter inference is slower:
mean/median/P95 0.4599/0.4528/0.4627 s versus base 0.3365/0.3366/0.3434 s. Adapter SHA
`6b4fce2e...735c` matches the frozen final contract. Peak allocated/reserved VRAM was
5,468,073,984/5,584,715,776 bytes; host wall 63.53 s, max RSS 5,848,468 KiB, zero swap/network and clean shutdown.

Both 70-frame HD videos and trajectory pass hash/media/visual checks. This pair demonstrates typical-state control
path acceleration, not faster inference or superior reliability. The frozen 43/45 environment non-regression FAIL
remains. Next is init44 base success then adapter timeout, strictly presentation-only with no tuning or final rerun.

## 23. P0-HD OpenVLA frozen-base init44 v2 PASS (latest; 2026-08-28)

Frozen base init44 execution/policy success PASSed at exactly 68 steps/reward 1.0. Load was 4.257 s; inference
total/mean/median/P95 23.425/0.3445/0.3390/0.3447 s (2.903 predictions/s). Peak allocated/reserved VRAM was
5,061,286,912/5,196,742,656 bytes; host wall 53.58 s, max RSS 5,776,412 KiB and zero swap/network. Frozen revision,
four shards, worker shutdown and GPU cleanup PASSed. Both 68-frame HD videos pass hash/media/visual QA.

The final formal P0-HD environment capture is the known frozen QLoRA init44 600-step timeout under a fresh
single-EGL v2 ID. It is presentation-only expected policy failure, not a new selection opportunity. Never tune from
it, replace the adapter or rerun Phase 3B final.

## 24. P0-HD QLoRA init44 EXPECTED_EXCEPTION and completion (latest; 2026-08-28)

Execution PASSed while the frozen policy outcome matched the preregistered exception: QLoRA init44 timed out at
600 steps with reward 0.0 and no success. Load was 5.189 s; inference total/mean/median/P95 was
291.498/0.4858/0.4843/0.5036 s (2.058 predictions/s). Peak allocated/reserved VRAM was
5,468,073,984/5,584,715,776 bytes; host wall 407.17 s, max RSS 5,872,152 KiB and zero swap/network. Frozen base
revision, adapter SHA `6b4fce2e...735c`, four shards, 600 predictions and cleanup all PASSed. The 600-frame HD videos
and trajectory pass hash/media/visual checks.

P0-HD is COMPLETE. A CPU-only offline step used zero environment steps/predictions to create two 1920x1080 paired
videos: init40 base 221 versus QLoRA 70 (both success), and init44 base 68 success versus QLoRA 600 timeout. Outputs:
`outputs/presentation/p0_hd_comparisons_v1_20260828`; manifest:
`results/p0_hd_comparisons_v1_manifest.yaml`. Frozen Phase 3 results remain base 45/45 and QLoRA 43/45; no tuning,
checkpoint substitution or final rerun is authorized.

## 25. P0-Breadth 5-task/15-episode protocol static PASS (latest; 2026-08-28)

To address the single-task portfolio limitation, a frozen OpenVLA-base LIBERO-Goal breadth gate is statically ready.
It preregisters task IDs `[0,2,3,6,9]` covering drawer articulation, cabinet placement, a compound drawer-and-bowl
task, object-to-container placement and bottle-to-rack placement. Init states `[0,1,2]` are fixed for every task,
giving 15 episodes, seeds 20261001--20261015 and a 600-step cap. One persistent frozen-base load serves all tasks;
each environment is closed before the next task and every step uses hashed native-128 IPC.

Static syntax, task name/language mapping, checkpoint revision, episode count, output freshness and no-overwrite
contracts PASSed with zero CUDA calls, environment steps or predictions. Config:
`configs/eval/p0_breadth_openvla_base_5task15_v1.yaml`; evaluator:
`scripts/evaluate_p0_breadth_openvla_base.py`. This is a small presentation breadth set, not a complete benchmark;
no failed task/state may be removed after outcomes are visible.

## 26. P0-Breadth expanded to all 10 LIBERO-Goal tasks (latest; 2026-08-28)

At the user's request, the unexecuted five-task v1 plan is preserved but superseded. Formal v2 covers task IDs 0--9,
the entire LIBERO-Goal task catalog, with fixed init states `[0,1,2]` per task: 30 episodes and seeds
20261001--20261030. The evaluator remains one frozen-base model load with environment reconstruction/cleanup per
task and hashed native-128 IPC per step. Result paths and experiment ID are new.

V2 static syntax, all 10 task name/language mappings, 30-episode arithmetic, checkpoint contract, output freshness
and no-overwrite guards PASSed with zero CUDA/environment/predictions. Formal config:
`configs/eval/p0_breadth_openvla_base_10task30_v2.yaml`. This covers the task catalog but is still only three states
per task, not the official complete benchmark. All outcomes must remain in the aggregate and cannot drive task/init
replacement or selective reruns.

## 27. P0-Breadth 10-task/30-episode execution PASS (latest; 2026-08-28)

The frozen v2 execution completed all 30 episodes in 51:06 with host/Python exit 0. Policy strict success was 20/30
(66.67%, Wilson 95% 48.78%--80.77%), and every task had at least one success (10/10 catalog coverage). Per-task
counts for IDs 0--9 were `[2,2,3,1,3,2,1,3,1,2]/3`. Ten failures all reached the 600-step cap; no failure was
removed. The weakest 1/3 tasks were compound drawer+bowl (T3), cream-cheese-to-bowl (T6) and bowl-to-plate (T8).

There were 8,554 environment steps/predictions. Successful episodes used mean/median 127.7/101 steps; all-episode
capped mean/median was 285.13/133.5. Inference mean/median/P95 was 0.3346/0.3331/0.3491 s (2.989 predictions/s).
Peak allocated/reserved VRAM was 5,068,574,720/5,196,742,656 bytes; host max RSS 5,773,664 KiB, zero swap/network,
and worker/GPU cleanup PASSed. Frozen revision matched and all four shards loaded. Offline/training metrics do not
apply. Canonical result: `results/p0_breadth_openvla_base_10task30_v2.yaml`.

The offline 10-task bar chart/summary is PASS with zero environment/model work at
`results/p0_breadth_summary_v1_manifest.yaml`. HD selection is frozen to the first successful init per task:
`[0,1,0,2,0,0,1,0,0,0]`, with observed steps `[114,90,77,246,93,133,108,89,76,270]`. The runner is static PASS
with fresh paths and cannot rerun failures or change 20/30.

## 28. P0-Breadth 10-task HD recapture and montage COMPLETE (latest; 2026-08-28)

The fixed first-success selection `[0,1,0,2,0,0,1,0,0,0]` completed 10/10 strict-success 640px recaptures. Every
episode exactly matched its frozen source step count `[114,90,77,246,93,133,108,89,76,270]`, totalling 1,296
environment steps/predictions. Host exit was 0; wall/user/system time was 785.02/1089.10/110.11 s, CPU 152%, max
RSS 5,860,536 KiB, zero swap/network, and the GPU compute process was released. Median inference latency across
captures ranged 0.3315--0.3388 s, P95 0.3363--0.3457 s; maximum allocated/reserved VRAM was
5,068,574,720/5,196,742,656 bytes. Checkpoint integrity and clean worker shutdown passed for all captures.

All video hashes, codecs and frame counts verified. Task 6's original presentation encode is preserved but marked
noncanonical because of visible early H264 corruption; its intact clean master was recomposed offline. Multi-point
visual QA passed for the repaired video and 1920x1080, 20 fps, 64.8 s ten-task montage. Canonical manifests are
`results/p0_breadth_hd_10task_v1_manifest.yaml` and `results/p0_breadth_10task_montage_v1_manifest.yaml`; canonical
montage is `outputs/presentation/p0_breadth_10task_montage_v1_20260828/openvla_10task_success_montage_hd.mp4`.
P0-Breadth is COMPLETE. These selected success demonstrations do not change the frozen 20/30 aggregate or permit
failure replacement/reruns.

## 29. Phase 4A viewpoint camera gate static PASS (latest; 2026-08-28)

P1/Phase 4 has started with a camera/runtime gate, not a policy rollout. The frozen OpenVLA base stays unchanged and
will later receive exactly one image from either `agentview` or `eye_in_hand` under the same rotate-180 plus frozen
center-crop contract. Development tasks `[0,3,6,7,8]` and final tasks `[1,2,4,5,9]` are fixed before any eye-in-hand
outcome; each uses init states `[0,1,2]` and both views, giving 30 development and 30 final episodes. Development may
diagnose integration only; it cannot train/select a checkpoint, and final is one-shot after protocol freeze.

The static gate PASSed all 18 checks with zero CUDA, resets, environment steps or model predictions. Config/result:
`configs/eval/phase4a_openvla_viewpoint_camera_gate_v1.yaml` and
`results/phase4a_openvla_viewpoint_camera_gate_v1_static.yaml`. The next authorized host gate performs one reset,
zero steps and zero predictions, saves/hash-checks native 128 and direct 640 images for both views, and closes the
environment. It cannot establish viewpoint robustness, success or reward.

## 30. Phase 4A camera gate v1 FAIL; corrected v2 static PASS (latest; 2026-08-28)

The v1 host probe reset successfully but FAILed before image emission because the observation alias `eye_in_hand`
was reused as a direct MuJoCo camera name. The simulator reports the correct name as `robot0_eye_in_hand`. Exit was
1; wall/user/system 13.27/19.30/4.77 s, CPU 181%, max RSS 2,491,256 KiB, zero swap/network, zero environment steps,
zero predictions and clean env close. This is infrastructure failure, not policy evidence. Raw log/output are
preserved and indexed by `results/phase4a_openvla_viewpoint_camera_gate_v1_failure.yaml`.

V2 changes only that direct-render camera name, with new immutable targets. Observation alias, checkpoint, task
split, common preprocessing and 30+30 future protocol remain frozen. Its nine-check static gate PASSed with zero
CUDA/reset/step/prediction. Canonical v2 config/static result are
`configs/eval/phase4a_openvla_viewpoint_camera_gate_v2.yaml` and
`results/phase4a_openvla_viewpoint_camera_gate_v2_static.yaml`. Future host commands must use a child shell rather
than setting `errexit` in the user's interactive shell, so an experimental nonzero exit cannot close the terminal.

## 31. Phase 4A corrected camera gate v2 PASS (latest; 2026-08-28)

V2 completed with Python/host exit 0, one reset, zero environment steps/predictions and clean environment close.
Both native 128 and direct 640 agent/wrist views were uint8 RGB, nonconstant and hash-verified. Native/direct
cross-view MAE was 76.4615/77.4986; visual QA confirmed a clear third-person workspace view and a distinct clear
wrist-mounted top-down view. Internal/host wall was 7.95/12.17 s, user/system 19.45/4.48 s, CPU 196%, max RSS
2,486,184 KiB, 13 major faults, zero swap/network.

No model/checkpoint was loaded, so loss, offline validation, token accuracy, L1, gradients, inference, VRAM,
success and reward do not apply. Canonical result is `results/phase4a_openvla_viewpoint_camera_gate_v2.yaml`.
Phase 4A camera feasibility is complete. Phase 4B development rollout must be implemented and statically frozen
under new paths before any GPU/environment command; this result alone is not viewpoint robustness evidence.

## 32. Phase 4B paired viewpoint development static PASS (latest; 2026-08-28)

The development protocol freezes tasks `[0,3,6,7,8]`, init states `[0,1,2]` and views
`[agentview,eye_in_hand]`: 15 same-task/init/seed pairs and 30 episodes. Each episode constructs a fresh environment
explicitly bound to its init state; one persistent frozen-base worker serves all model calls. Only the observation
key changes between paired views. Primary metrics are strict success by view, Wilson intervals, paired outcome
categories and per-task coverage; step deltas apply only to both-success pairs. All 600-step failures remain.

Static validation PASSed 14 checks with zero CUDA/reset/step/prediction. It binds canonical revision
`fa5ae1e7509348889295bba8e08621d8b55e9baf`; Phase 4A's different unloaded revision string is now explicitly logged
as metadata-only correction in `results/phase4a_checkpoint_revision_metadata_correction.yaml`. Phase 4A camera
evidence is unaffected because no model was loaded. Phase 4B config/static result:
`configs/eval/phase4b_openvla_viewpoint_dev30_v1.yaml` and
`results/phase4b_openvla_viewpoint_dev30_v1_static.yaml`. Final tasks `[1,2,4,5,9]` remain untouched. No training,
checkpoint selection, result removal or selective rerun is authorized.

## 33. Phase 4B paired viewpoint development execution PASS / direct substitution NO-GO (latest; 2026-08-28)

All 30 paired development episodes executed successfully. Agentview achieved 10/15 strict success (Wilson 95%
41.71%--84.82%), five timeouts, 5/5 task coverage and reward sum 10. Eye-in-hand achieved 0/15 (Wilson upper
20.39%), 15 timeouts, 0/5 task coverage and reward sum 0. Paired outcomes were 0 both, 10 agent-only, 0 wrist-only
and 5 neither; the two-sided exact paired binomial/McNemar p-value is 0.001953. There are no both-success pairs, so
paired completion-step comparison is not applicable. This is systematic degradation on development, not final or
general viewpoint-invariance evidence.

The run used 13,663 environment steps/predictions. Load time was 4.60 s; inference mean/median/P95
0.3401/0.3404/0.3483 s, throughput 2.941 predictions/s. Peak allocated/reserved VRAM was
5,068,574,720/5,196,742,656 bytes; internal/host wall 4,962.71/4,966 s; max RSS 5,867,244 KiB, zero swap/network.
Exact revision and four shards matched; worker shutdown PASSed at 13,663 predictions. Offline/training/gradient
metrics do not apply.

Decision: `NO_GO_DIRECT_EYE_IN_HAND_SUBSTITUTION`. Final tasks `[1,2,4,5,9]` remain untouched and are not authorized
for this failed hypothesis. Any continuation must first preregister a new ego-view adaptation/alignment hypothesis
using development evidence only; no final-driven tuning or selective rerun is allowed. Canonical result:
`results/phase4b_openvla_viewpoint_dev30_v1.yaml`.

CPU-only visualization added zero steps/predictions. Canonical visual-QA PASS assets are indexed by
`results/phase4b_viewpoint_dev_summary_v2_manifest.yaml`; v1 is preserved noncanonical because its footer overlapped
axis labels. `scripts/verify_phase4b_assets.py` is the read-only integrity entrypoint.

## 34. Phase 4C wrist adaptation data/training static gates PASS (latest; 2026-08-28)

After Phase 4B direct substitution NO-GO, the next hypothesis is a single-task development-only rank-8 QLoRA on
native wrist demonstration images. Existing turn-on-stove data is sufficient without download: the unchanged 40/10
episode split contains 3,571/889 train/validation frames. Wrist RGB shape/dtype/dynamic range, finite 7D actions,
dataset SHA and final-task boundary PASSed; mean episode-level cross-view MAE is 76.02. A deterministic wrist-only
20-train/10-val fixture was exported and hashed with zero CUDA/environment/model work.

The initial training static checker FAILed solely because its expected reused-trainer SHA was mistyped; zero CUDA or
training occurred and the result is preserved. Corrected v2 binds actual trainer SHA `e58f91aa...a178` and PASSed
all 12 checks. The host gate reuses the exact Phase 3B rank-8/NF4/AdamW 20-update machinery via config-only wrapper;
expected peak reserved VRAM is about 7.13 GB. Offline loss/token accuracy remain distinct from environment success,
the smoke cannot select a checkpoint, and final tasks stay untouched.

## 35. Phase 4C wrist QLoRA 20-update smoke PASS with mixed offline signal (latest; 2026-08-28)

Training execution PASSed all 20 updates. All loss/gradient values and 879 adapter tensors were finite; optimizer/RNG
state is complete at step 20 with 852 optimizer entries. Last-ten train loss averaged 11.585 versus 13.311 for the
first ten (-12.97%), while pre-clip gradient norm averaged 438.8 (361.6--533.1).

Fixed wrist val10 loss improved 13.704 to 9.279 (-32.29%), but action-token accuracy regressed from 16/70 to 13/70
(-4.29 percentage points). Decision is `PASS_ENGINEERING_MIXED_OFFLINE_SIGNAL_NO_CHECKPOINT_SELECTION`; neither
metric is success or reward. Load/adapter/train+validation time was 5.69/0.35/21.34 s, host wall 31.96 s. Peak
allocated/reserved VRAM was 6,866,078,208/7,065,305,088 bytes; host max RSS 5,776,904 KiB, zero swap/network.
Adapter, optimizer/RNG and five checkpoint files passed hash/CPU integrity.

Canonical result: `results/phase4c_openvla_wrist_qlora_train_smoke_v1.yaml`. No environment rollout or checkpoint
selection is authorized. The next required gate is paired base-versus-adapter full validation on all 889 frozen
wrist frames, under a new static contract; final tasks remain untouched.

## 36. Phase 4D paired wrist fullval889 static PASS (latest; 2026-08-28)

All 889 frames from frozen validation episodes `[0,5,12,21,29,34,40,42,43,45]` were exported from native
`obs/eye_in_hand_rgb` with identical labels into a 15 MB hashed fixture. Frozen base and Phase 4C step-20 adapter
configs share the exact fixture, split and canonical revision. Adapter SHA and both previously frozen Phase 3B
fullval implementation hashes are bound through config-only wrappers with fresh outputs.

Static validation PASSed 11 checks with zero CUDA/frame evaluation/environment work. Execution order is fixed:
base first, audit its result and resources, then adapter. Fullval reports loss, action-token accuracy and decoded L1
only; none are success/reward or automatic rollout authorization. Canonical static result:
`results/phase4d_openvla_wrist_fullval889_static.yaml`.

## 37. Phase 4D frozen-base wrist fullval889 PASS (latest; 2026-08-28)

Base processed all 889 frames/6,223 action tokens with finite metrics: mean loss 16.6834, token accuracy 13.0805%
and decoded L1 0.28796. Load/validation was 5.68/201.19 s, 4.419 frames/s, median/P95 forward
0.2235/0.2293 s. Peak allocated/reserved VRAM was 5,167,181,312/5,335,154,688 bytes; host wall 209.32 s, max RSS
5,768,068 KiB, zero swap/network. Revision/fixture/output integrity PASSed; no training/environment/selection occurred.

Canonical result: `results/phase4d_openvla_wrist_base_fullval889_v1.yaml`. This is teacher-forced offline evidence,
not success/reward. The step-20 adapter is now authorized only for the already-frozen same-fixture offline run.

## 38. Phase 4D wrist adapter fullval889 PASS / offline adaptation PASS (latest; 2026-08-28)

The step-20 adapter completed the exact same 889 frames/6,223 tokens with finite rows and all hashes intact. Relative
to frozen base, loss changed 16.6834→9.0479 (-45.77%), token accuracy 13.0805%→20.8099% (+7.73 percentage points;
814→1,295 correct tokens), and continuous L1 0.28796→0.19459 (-32.43%). Loss/L1 improved on 884/756 of 889
frames, and all 10 validation episodes improved on all three aggregate metrics.

Median/P95 latency increased 15.69%/15.43%, throughput fell 13.28%, and allocated/reserved VRAM increased
5.82%/4.68%. Adapter host wall/max RSS was 240.92 s/5,846,720 KiB with zero swap/network. No training,
environment execution or checkpoint selection occurred during this run.

Decision: `OFFLINE_ADAPTATION_PASS_ENVIRONMENT_RECOVERY_UNKNOWN`. This teacher-forced result is not success or
reward. Step-20 may only become the fixed candidate in a newly preregistered development rollout; final tasks
`[1,2,4,5,9]` remain untouched. Canonical comparison and visual manifest are
`results/phase4d_openvla_wrist_step20_vs_base_fullval889_paired.yaml` and
`results/phase4d_wrist_fullval_summary_v1_manifest.yaml`.

## 39. Phase 4E task-7 wrist-adapter development gate static PASS (latest; 2026-08-28)

The next closed-loop gate is intentionally limited to the adapter's training task. Immutable Phase 4B evidence
already supplies frozen-base eye-in-hand outcomes for task 7, init `[0,1,2]`, seeds
`[20261120,20261121,20261122]`: 0/3, each a 600-step timeout. Phase 4E will not rerun base; it runs the fixed
step-20 adapter once on each identical state/seed and retains every failure.

The preregistered rule is 0/3=`NO_GO_NO_CLOSED_LOOP_RECOVERY`, 1–2/3=`DEVELOPMENT_RECOVERY_SIGNAL_PARTIAL`, and
3/3=`DEVELOPMENT_RECOVERY_SIGNAL_STRONG`. Static validation PASSed 12/12 checks with zero CUDA/reset/step/prediction.
No outcome automatically authorizes final tasks, checkpoint tuning, selective reruns or cross-task ego-view claims.
Canonical config/result are `configs/eval/phase4e_openvla_wrist_adapter_dev3_v1.yaml` and
`results/phase4e_openvla_wrist_adapter_task7_dev3_v1_static.yaml`.

## 40. Phase 4E execution PASS / wrist closed-loop recovery NO-GO (latest; 2026-08-28)

All three preregistered adapter episodes completed, but success remained 0/3. Init states 0, 1 and 2 each ran 600
steps with sum/max reward 0 and no termination; paired against immutable base evidence, all three outcomes are
`neither_success`. Decision is `NO_GO_NO_CLOSED_LOOP_RECOVERY`. Phase 4D's strong teacher-forced improvements did
not recover task-7 wrist closed-loop control.

The run produced 1,800 finite in-range actions and shut down cleanly at 1,800 predictions. Mean/median/P95 inference
was 0.4876/0.4880/0.5033 s; mean was 43.02% slower than matching Phase 4B base wrist episodes. Peak allocated/
reserved VRAM was 5,468,073,984/5,584,715,776 bytes; internal/host wall 931.18/937.42 s; host max RSS 5,774,904
KiB, zero swap/network. Revision, adapter, config, trajectory and raw-log integrity PASSed.

No offline loss, gradients, training or checkpoint selection apply to this rollout. Do not tune, switch checkpoint,
selectively rerun or execute final tasks from these failures. Canonical result:
`results/phase4e_openvla_wrist_adapter_task7_dev3_v1.yaml`.

## 41. Phase 4 viewpoint robustness closeout COMPLETE / NO-GO (latest; 2026-08-28)

Phase 4 is closed as a coherent negative-result study: camera/runtime feasibility PASS; direct frozen-base wrist
substitution NO-GO (agent 10/15 versus wrist 0/15); native-wrist rank-8 QLoRA offline adaptation PASS (loss
-45.77%, token accuracy +7.73 pp, L1 -32.43%); task-7 closed-loop recovery NO-GO (base 0/3, adapter 0/3, all
600-step timeouts). The final split was never touched.

The canonical report-safe conclusion is that teacher-forced wrist fitting did not establish closed-loop viewpoint
transfer. The hypothesis is frozen against retuning, checkpoint switching and selective reruns. A future study must
preregister a genuinely new mechanism. Closeout document/visual manifest are
`docs/phases/phase_4_viewpoint_closeout.md` and `results/phase4_viewpoint_closeout_v1_manifest.yaml`; portfolio,
resume, README and roadmap now reflect Phase 4 COMPLETE/NO-GO rather than “not started.”
`scripts/verify_phase4_closeout_assets.py` PASSes 17/17 read-only checks across four canonical manifests and seven
figure artifacts with zero CUDA/environment/model work.

## 42. P0 public project report and unified reproduction index COMPLETE (latest; 2026-08-28)

`docs/project_report.md` now provides a reviewer-facing synthesis of architecture, PushT, ACT-on-LIBERO, frozen
OpenVLA, QLoRA, ten-task breadth, viewpoint NO-GO, media, limitations and evidence semantics. `docs/repro_index.md`
maps every headline claim to its canonical result, config/protocol, manifest and verifier, and restates fresh-run and
no-overwrite requirements.

`results/p0_project_closeout_v1_manifest.yaml` hash-indexes the report, reproduction index, portfolio, resume,
ten-task montage and Phase 4 evidence-chain figure. `scripts/verify_project_closeout.py` PASSes 16/16 checks: six
indexed hashes, four canonical manifests, four verifier entries and all 35 local references in the reproduction
index. This was CPU-only documentation work with zero CUDA/environment/model execution. P0 reporting and
reproduction closeout is COMPLETE.

## 43. Phase 5A small-world-model temporal data contract PASS (latest; 2026-08-28)

CPU-only audit of the existing turn-on-stove HDF5 PASSed 12/12 data checks. The frozen 40/10 episode split contains
3,571/889 frames; train transition windows for H=1/2/4/8 are 3,531/3,491/3,411/3,251 and validation windows are
879/869/849/809. Agent/wrist RGB, 7D actions, 9D robot state, 79D simulator state, rewards and terminal markers are
aligned; all action dimensions are finite and nonconstant. No window crosses an episode.

The validation copy-frame baseline shows materially different difficulty by camera. Agentview MAE grows
0.769→2.402 from H1 to H8, while wrist grows 6.278→24.594; corresponding PSNR changes 33.28→26.55 dB versus
22.39→14.85 dB. These are non-learned temporal baselines, not world-model performance, reward prediction or action
ranking. Decision: `DATA_CONTRACT_PASS_BASELINE_IMPLEMENTATION_ALLOWED`; no training is authorized.

Canonical config/result/figure manifest are `configs/dataset/phase5a_libero_world_model_contract_v1.yaml`,
`results/phase5a_libero_world_model_data_contract_v1.yaml` and `results/phase5a_world_model_data_v1_manifest.yaml`.
The read-only verifier PASSes 5/5 hashes/checks with zero CUDA/environment/model work.

## 44. Phase 5B small-world-model architecture static PASS (latest; 2026-08-28)

A shared 4-stage convolutional encoder/decoder plus parameter-matched action-free and action-conditioned residual
latent dynamics is implemented. On real episode-0 data, both produced finite `[2,8,3,128,128]` open-loop RGB
rollouts and finite gradients. Action-free/conditioned sizes are 1,284,355/1,292,803 parameters (0.66% difference),
and recorded versus zero actions changed conditioned output, confirming a connected action path.

Static validation PASSed 16/16 checks. Random-weight losses 0.3358/0.3404 and action sensitivity are implementation
checks only, not learned performance. Internal/host CPU wall was 0.90/2.37 s, max RSS 968,836 KiB, zero swap/network,
no CUDA, optimizer step or environment work. Decision is
`ARCHITECTURE_PASS_TRAINING_SMOKE_IMPLEMENTATION_ALLOWED`; formal training remains unauthorized. Canonical result:
`results/phase5b_small_world_model_static_v1.yaml`.

## 45. Phase 5C paired world-model training smoke static PASS (latest; 2026-08-29)

A deterministic reference-only fixture freezes 256 train and 128 validation H4 windows from the unchanged Phase 5A
40/10 episode split. Action-free and action-conditioned models each receive 100 updates with identical batch order
and identical encoder/decoder initialization, then identical H1–4 pre/post validation plus copy-frame comparison.

Static validation PASSed 14/14 checks with zero CUDA/forward/training/environment work. Both checkpoints and optimizer
states must be retained; no checkpoint is selected. The run can establish engineering stability and an offline
response only—not world-model quality, conditioned superiority, planning/reward utility or environment success.
Formal training remains forbidden. Canonical config/static result are
`configs/train/phase5c_world_model_train_smoke_v1.yaml` and
`results/phase5c_world_model_train100_pair_v1_static.yaml`.

## 46. Phase 5C execution engineering PASS / offline baseline NO-GO (latest; 2026-08-29)

Both paired models completed 100 finite updates with identical batches and encoder/decoder initialization. First-ten
to last-ten loss fell 72.86% action-free and 72.50% conditioned; finite pre-clip gradients stayed within
0.0364–0.3222/0.0365–0.3193. Both model/optimizer checkpoints CPU-load completely at step 100.

They fail the frozen offline baseline. H4 MAE is 11.526 action-free and 11.722 conditioned versus copy-frame 1.732.
Conditioned is 1.71–1.78% worse than action-free at every H1–4 and 576.96% worse than copy at H4. Thus optimization
works, but no action benefit or competitive dynamics is established. Decision:
`ENGINEERING_PASS_OFFLINE_BASELINE_NO_GO`; no checkpoint selection/formal training/planning/environment work.

Internal/host wall was 2.65/4.53 s, 75.47 total updates/s, peak allocated/reserved VRAM
141,130,240/165,675,008 bytes, max RSS 1,775,364 KiB, zero swap/network. Canonical result/visual manifest are
`results/phase5c_world_model_train100_pair_v1.yaml` and `results/phase5c_world_model_smoke_v1_manifest.yaml`.

## 47. Phase 5D copy-preserving residual mechanism static PASS (latest; 2026-08-29)

Phase 5C's full-frame predictor is frozen as a negative baseline rather than extended. A new bounded residual model
predicts `clamp(image_t + residual, 0, 1)` and zero-initializes its residual head. Both action-free and
action-conditioned variants therefore reproduce the current frame exactly at initialization (max absolute
difference 0) while retaining finite backward paths and a measurable conditioned latent action effect.

CPU static validation PASSed 12/12. H4 rollout shapes were `[2,4,3,128,128]`; model sizes were
1,284,355/1,292,803 parameters. The original 40 training episodes are now nested into frozen 32-train/8-development
sets; the old 10 validation episodes are explicitly `prior_exposed_confirmation`, not untouched final evidence.
No training or confirmation execution was authorized. Canonical result:
`results/phase5d_residual_world_model_static_v1.yaml`.

## 48. Phase 5E residual paired development smoke static PASS / host pending (latest; 2026-08-29)

A deterministic H4 fixture freezes 256 training windows from the 32-train set and 128 development windows from the
8-development set. It contains none of the 10 prior-exposed confirmation episodes. Residual action-free and
conditioned models will receive 100 updates each with identical batches and shared encoder/decoder initialization.

The preregistered GO rule requires conditioned post-training H4 MAE to beat both copy-current-frame and residual
action-free. A GO would still not authorize confirmation evaluation, checkpoint selection or formal training.
Static v1 identified that the trainer did not emit the declared canonical YAML; no GPU/model work occurred and v1
is retained. After packaging-only correction, execution-authorizing static v2 PASSed 18/18 with zero torch/CUDA
import, model prediction, optimizer step or environment step.
Canonical config/static result are `configs/train/phase5e_residual_world_model_dev_smoke_v1.yaml` and
`results/phase5e_residual_world_model_dev100_pair_v1_static_v2.yaml`. GPU execution is pending explicit host action.

## 49. Phase 5E execution PASS / action conditioning development NO-GO (latest; 2026-08-29)

Both residual models completed 100 finite updates with identical batches and shared encoder/decoder initialization.
The execution and checkpoint integrity PASSed, but the preregistered hypothesis failed: development H4 pixel MAE
was copy 1.59125, residual action-free 1.59223 and residual action-conditioned 1.59303. Conditioned was 0.0506%
worse than free and 0.1124% worse than copy. First-10 to last-10 loss changed only -3.73%/-3.67%; gradients were
finite and bounded below 0.137 pre-clip.

Both step-100 model/optimizer checkpoints CPU-load completely. Internal/host wall was 2.825/6.23 s, total internal
throughput 70.80 updates/s, peak allocated/reserved VRAM 133,392,896/159,383,552 bytes, host max RSS 1,814,380
KiB, zero swap/network. No confirmation episode, environment step, success/reward, checkpoint selection or planning
evaluation occurred. Decision: `DEVELOPMENT_NO_GO_ACTION_CONDITIONING`; do not extend training or inspect the
prior-exposed confirmation set. Phase 5 stops this mechanism and the next main line is Phase 6 Feagine interface and
safety-contract work. Canonical result/manifest are `results/phase5e_residual_world_model_dev100_pair_v1.yaml` and
`results/phase5e_residual_world_model_dev_summary_v1_manifest.yaml`.

The current report/reproduction index is hash-frozen separately in
`results/project_closeout_v2_phase5e_manifest.yaml`; the historical P0 v1 manifest is preserved unchanged.
`scripts/verify_project_closeout_v2.py` PASSes 12/12 checks including 40 local reproduction references.

## 50. Phase 6A Feagine generic bridge static PASS / A03 binding blocked (latest; 2026-08-29)

The workspace contains no Feagine A03 MJCF, calibrated section geometry, actuator mapping or existing PCC API. A
provisional single-section extensible-PCC reference therefore implements only the software boundary: LIBERO-style
7D delta schema, explicit frame transform, translation rate limit, gripper clamp, workspace checks, closed-form
position FK/IK and fail-closed rejection of NaN, unsupported rotation and unreachable targets.

Static v1/v2 are preserved `EXPECTED_EXCEPTION` packaging failures from NumPy booleans; v3 is a preserved 13/14
FAIL caused by a test fixture that ignored the rate limiter and landed exactly on the valid workspace boundary. The
corrected fresh-ID v4 PASSed 14/14. Across 256 deterministic FK→IK→FK samples, maximum/mean position error was
8.58e-16/2.95e-17 m. There were zero CUDA/model/environment/hardware operations.

This establishes a generic numerical/safety contract only, not A03 fidelity or simulation transfer. Phase 6B is
blocked on locating/providing the actual A03 asset and definitions; do not invent them or launch MuJoCo first.
Canonical result/doc/verifier: `results/phase6a_feagine_bridge_static_v4.yaml`,
`docs/phases/phase_6a_feagine_bridge_static.md`, `scripts/verify_phase6a_assets.py`.

Current report/reproduction hashes are frozen in `results/project_closeout_v3_phase6a_manifest.yaml`; older closeout
manifests remain historical snapshots.

## 51. Phase 6B A03 Type-2 distribution asset contract PASS / runtime install pending (latest; 2026-08-29)

The user-provided assets are under the project-local `feagine-simulation/dist` directory. Distribution v1.0.1
contains core 0.1.0 revision `2c59c71`, MuJoCo wheel 0.1.2 and the sole preset `a03_type_2`. The MuJoCo wheel SHA-256
is `0c4bfe832457a6fb4238e1f1f743a5a5334df3399a7b5c24d3493803259f0650`.

No-install archive audit PASSed 19/19. The actual contract is three fixed 0.16 m sections, five samples/section,
theta `[0,2.356194]`, phi `[-3.141593,3.141593]`, 0.0235 m backbone radius, 0.153 m tool offset and 0.005 s
MuJoCo timestep. MJCF has 52 joints and 51 actuators: 48 capsule proxy joint/actuator pairs plus grasper rotation
and two finger pairs. The bridge now emits core-compatible target position + unit tip direction and maps LIBERO
`-1=open,+1=close` to wrapper `0=open,1=close`; uncalibrated rotations fail closed.

Packaging caveat: helper code defaults to preset `a03`, but this wheel has only `a03_type_2`; all calls must pass
the explicit preset. `evla-core` has MuJoCo 3.8.1 but neither `feagine_mujoco` nor `pyfeagine_sim_core`; no install,
MJCF load or simulation step occurred. Decision `ASSET_CONTRACT_PASS_RUNTIME_INSTALL_PENDING`. Canonical evidence:
`results/phase6b_feagine_a03_asset_contract_v1.yaml`, `docs/phases/phase_6b_feagine_a03_asset_contract.md`.

Current report/reproduction hashes are frozen in `results/project_closeout_v4_phase6b_manifest.yaml`.

## 52. Phase 6C isolated runtime zero-step gate static PASS / host pending (latest; 2026-08-29)

The next gate is implemented and static validation PASSed 6/6. It requires a new isolated `evla-feagine-v1`
environment, builds the supplied core extension and installs the immutable local MuJoCo wheel. Runtime validation
must use explicit preset `a03_type_2`, load the 52-joint/51-actuator MJCF, bind the wrapper, exercise core FK/IK and
write finite controller targets while keeping `data.time == 0`.

The runner contains no `mj_step` call and forbids GUI, GPU, policy inference and hardware work. Static validation
performed no import/install/build/MJCF load. Canonical protocol/static result:
`configs/task/phase6c_feagine_a03_runtime_zero_step_v1.yaml` and
`results/phase6c_feagine_a03_runtime_zero_step_v1_static.yaml`.

## 53. Phase 6C runtime v1 FAIL from named-joint expectation / v2 recovery static PASS (latest; 2026-08-29)

The isolated environment, Eigen dependency, core build/install and local MuJoCo wheel install all succeeded.
Runtime v1 then returned structured 10/11 FAIL solely because the protocol expected 52 runtime joints. The archive
contains 52 XML `<joint>` elements only when the unnamed `<default><joint>` template is included; MuJoCo correctly
instantiates 51 named joints and 51 actuators. Required first/terminal capsule and all grasper names were present.

All other runtime evidence PASSed: explicit `a03_type_2`, constants, 5 ms timestep, finite control writes, core FK,
and IK success in 5 iterations with 74.67 μm position error and 1.35e-4 direction error. `data.time` remained
0.0→0.0; zero environment steps, GPU/model predictions and hardware commands. Runner host wall was 0.31 s, max RSS
247,724 KiB, no swap/network. The v1 result/output/raw log are preserved.

Correction `results/phase6b_mjcf_joint_count_correction.yaml` records 51 named runtime joints. Fresh v2 recovery
changes only that expectation, reuses the installed environment/build without reinstall/rebuild, and static PASSed
9/9. No simulation step is authorized until v2 runtime PASS.

## 54. Phase 6C runtime v2 zero-step PASS / Phase 6D short dynamics static PASS (latest; 2026-08-29)

The corrected v2 gate PASSed 11/11 with MuJoCo 3.8.1, explicit `a03_type_2`, 51 joints/51 actuators and 5 ms
timestep. Core IK succeeded in 5 iterations with 74.67 μm position and 1.35e-4 direction error. All controls were
finite and `data.time` remained exactly 0.0→0.0. Internal/host wall was 0.115/0.26 s; max RSS 247,972 KiB, zero
swap/network. This is runtime/interface evidence only, not dynamics or task success.

Canonical manifest/verifier are `results/phase6c_feagine_a03_runtime_summary_v1_manifest.yaml` and
`scripts/verify_phase6c_assets.py` (6/6 PASS). Phase 6D now freezes a one-second headless dynamics smoke: exactly
200 steps at 5 ms with small section/gripper/rotation commands, finite-state/joint-limit/tip-motion checks, no render,
IK target, reward, success, policy or hardware. Its CPU-only static authorization PASSed 7/7; host execution pending.

Current report/reproduction hashes are frozen in `results/project_closeout_v5_phase6c_manifest.yaml`.

## 55. Phase 6D short dynamics v1 joint-limit FAIL / diagnostic v2 static PASS (latest; 2026-08-29)

The one-second headless actuator smoke executed all 200 steps but returned 6/7 FAIL on `joint_limits`. State/control
remained finite, simulated time was 1.0000000000000007 s, terminal proxy moved 0.067034 m, max absolute qvel/control
were 0.5921/0.2, and contacts remained zero. Internal/host wall was 0.118/0.27 s (1691.7 steps/s), max RSS 241,092
KiB, zero swap/network. There was no reward, success, policy, rendering, GPU or hardware work.

v1 did not record the violating joint identity or magnitude, so it cannot justify tolerance relaxation or command
changes. The result/log/output are hash-preserved in `results/phase6d_feagine_a03_short_dynamics_v1_manifest.yaml`.
A fresh v2 diagnostic repeats the exact 200-step command and records per-joint extrema/penetration, with preregistered
material thresholds 1e-3 rad hinge and 1e-4 m slide. Static authorization PASSed 8/8; target reaching remains blocked.

Current report/reproduction hashes are frozen in `results/project_closeout_v6_phase6d_v1_manifest.yaml`.

## 56. Phase 6D per-joint limit diagnostic v2 PASS (latest; 2026-08-30)

The identical 200-step/1-second bounded dynamics protocol executed successfully. All state remained finite and
simulation time was exact. Of 51 limited joints, only slide joint `feagine_grasper_joint_1` crossed its nominal
0.005 m upper bound, reaching 0.0050117734 m. Its maximum penetration was 1.17734e-5 m (11.77 micrometres), 11.77%
of the preregistered 1e-4 m material threshold; maximum hinge penetration was zero. The decision is
`SOFT_LIMIT_PENETRATION_WITHIN_DIAGNOSTIC_THRESHOLD`.

Internal/host wall time was 0.118/0.34 s, host max RSS 240,704 KiB, with zero swap and network traffic. CUDA was not
initialized; model predictions, rewards, successes and hardware commands were absent. This explains the v1 strict
limit failure but does not revise it. It also does not authorize target reaching, policy transfer or hardware work.
Canonical result/config are `results/phase6d_feagine_a03_limit_diagnostic_v2.yaml` and
`configs/task/phase6d_feagine_a03_limit_diagnostic_v2.yaml`.

## 57. Phase 6E bounded A03 move-to-target PASS (latest; 2026-08-30)

The first target-reaching development gate is implemented and static validation PASSed 10/10 without importing
MuJoCo or executing an environment step. A single reachable target is generated from the same PCC core using small
section angles `[0.06,0,0.04,0,0.02,0]`; core IK then drives the packaged MJCF for exactly 400 steps at 5 ms.

Before runtime results, PASS is frozen as target displacement 0.02–0.08 m, terminal tool-position error at most
0.015 m, at least 80% error reduction, finite state, exact two-second time and no material limit violation under
Phase 6D thresholds. This contains no VLA, image, arbitrary target, reward, obstacle, rendering or hardware work.
Canonical config/static result: `configs/task/phase6e_feagine_a03_move_to_target_smoke_v1.yaml` and
`results/phase6e_feagine_a03_move_to_target_smoke_v1_static.yaml`.

Host runtime then PASSed all 9 gates. Initial target error was 53.896 mm; after exactly 400 steps/two simulated
seconds, terminal error was 9.504 mm and progress was 82.366%. Core IK converged in four iterations with 1.064 μm
position and 7.55e-7 direction error. Maximum hinge penetration was zero, slide penetration 1.29e-12 m and contacts
zero. Internal/host wall was 0.137/0.32 s, max RSS 244,084 KiB, zero swap/network. No model predictions, reward or
hardware commands occurred. Canonical runtime evidence is indexed by
`results/phase6e_feagine_a03_move_to_target_smoke_v1_manifest.yaml` and `scripts/verify_phase6e_assets.py`.

## 58. Phase 6F explicit Cartesian target dev4 0/4 FAIL (latest; 2026-08-30)

Phase 6F removes the core-generated-target shortcut by freezing four explicit local Cartesian deltas: ±35 mm X and
±35 mm Y. Every target uses a fresh A03 reset and 400 steps at 5 ms. Per-target PASS requires successful IK, finite
state, exact time, terminal error ≤15 mm, progress ≥70% and no material limit violation; aggregate PASS is strictly
4/4. All failures must remain visible and targets cannot be replaced after outcomes.

Static validation PASSed 11/11 without importing MuJoCo or executing environment steps. This remains a small local
development gate with no VLA, image, reward, obstacle/contact objective, final split or hardware command. Canonical
config/static result: `configs/task/phase6f_feagine_a03_external_targets_dev4_v1.yaml` and
`results/phase6f_feagine_a03_external_targets_dev4_v1_static.yaml`.

The host run already completed on 2026-08-30 and returned `EXTERNAL_TARGETS_DEV4_FAIL`, 0/4. ±X both failed core
IK after one iteration with 36.27 mm solver position error and essentially zero tip motion. ±Y both passed numerical
IK in 16 iterations, but their MJCF terminal errors expanded from 35 mm to 241.63/271.86 mm. Every episode retained
finite state, exact two-second time, no material limit violation and zero contacts. Host/internal wall was
0.56/0.392 s, max RSS 371,580 KiB, zero swap/network; no model prediction, reward or hardware command occurred.

The failure is therefore in external target→IK section representation→MJCF controller mapping, not general dynamics
stability. Preserve v1 unchanged. The only justified next gate is a same-target diagnostic recording raw solved,
post-clamp applied section angles, actuator targets, and core-predicted versus MJCF tip motion. Canonical evidence:
`results/phase6f_feagine_a03_external_targets_dev4_v1_manifest.yaml`, `scripts/verify_phase6f_assets.py`.

## 59. Phase 6G same-target mapping diagnostic complete (latest; 2026-09-01)

Phase 6G repeats the immutable Phase 6F four targets and 1600-step budget with no mapping correction. It records raw
IK section angles, post-preset applied angles, all 48 PCC actuator targets, core-predicted and MJCF-actual tool
displacements, pairwise direction cosines and displacement gains. Diagnostic execution PASS only requires complete
finite records and exact timing; it cannot revise the Phase 6F 0/4 result.

Static validation PASSed 10/10 with zero runtime imports/environment steps. No sign conversion, canonicalization,
target replacement, VLA, reward, rendering or hardware is authorized. Canonical config/static result:
`configs/task/phase6g_feagine_a03_mapping_diagnostic_v1.yaml` and
`results/phase6g_feagine_a03_mapping_diagnostic_v1_static.yaml`.

Runtime diagnostic completed all four targets with finite state and exact time. ±X failed from the all-zero straight
IK start and emitted zero section angles/48 controls. +Y raw theta `[0.4250,-0.6264,0.2013]` became
`[0.4250,0,0.2013]`; -Y raw theta `[-0.4247,0.6257,-0.2010]` became `[0,0.6257,0]`. This direct preset clamp
destroyed the signed-theta solutions. Yet after clamp, core-predicted and MJCF-actual displacement cosine was
0.99941/0.99927, showing the actuator/dynamics layer closely followed the applied configuration.

Internal/host wall was 0.418/0.74 s, max RSS 379,084 KiB, zero swap/network; no model prediction, reward or hardware
command occurred. Phase 6F remains 0/4. The evidence supports—but has not executed—the equivalent representation
`theta<0 → abs(theta), phi → wrap(phi+pi)` plus frozen symmetric nonzero multi-start seeds for ±X. Canonical evidence:
`results/phase6g_feagine_a03_mapping_diagnostic_v1_manifest.yaml`, `scripts/verify_phase6g_assets.py`.

## 60. Phase 6H signed-theta equivalence PASS / multi-start target solve FAIL (latest; 2026-09-01)

The CPU-only core gate returned 4/5 FAIL with no MuJoCo import or environment step. The proposed negative-theta
canonicalization (`abs(theta)`, `wrap(phi+pi)`) preserved FK across 256 seeded random configurations plus the four
Phase 6G raw solutions: maximum position/direction differences were 9.22e-8 m and 4.28e-7.

Eight frozen symmetric nonzero seeds escaped the ±X zero-start singularity, but none solved any of the four original
targets within the unchanged 0.1 mm position tolerance. Best canonical FK residuals were consistently
1.592–1.615 mm with direction errors about 2.6–3.5e-5. This indicates that adding 35 mm lateral displacement while
preserving the straight arm's near-maximum axial extension and tip direction is likely geometrically infeasible for
fixed-length sections. Do not relax tolerance or promote approximate solutions. The next gate must quantify required
axial retraction/reachability before any MJCF recovery. Canonical evidence:
`results/phase6h_feagine_signed_theta_multistart_static_v1_manifest.yaml`, `scripts/verify_phase6h_assets.py`.

## 61. Phase 6I axial-retraction reachability scan recovered PASS (latest; 2026-09-01)

The CPU-only scan retained 35 mm lateral magnitude, straight tip direction and 0.1 mm IK tolerance while freezing
21 axial retractions from 0 to 5 mm at 0.25 mm spacing. Across 672 IK attempts, all ±X/±Y directions first became
reachable at 1.50 mm retraction with zero directional spread. At 1.25 mm best residual remained about 0.347 mm;
at 1.50 mm it was about 0.0982 mm.

v1 recorded 5/6 FAIL solely because decimal YAML floats were compared by exact list equality to multiplication-
generated Python floats. Its full evidence is immutable. Packaging-only v2 performed zero new FK/IK calls and
PASSed 10/10 by checking the same 21 grid values within 1e-12 m plus hashes/count/endpoints. This establishes a
geometrically justified future target contract, not MJCF recovery or policy transfer. Canonical evidence:
`results/phase6i_feagine_a03_reachability_summary_v1_manifest.yaml`, `scripts/verify_phase6i_assets.py`.

## 62. Phase 6J reachability-corrected MJCF dev4 static PASS / host pending (latest; 2026-09-01)

Phase 6J preserves the original four ±X/±Y 35 mm directions but adds the uniformly measured 1.50 mm axial
retraction. It freezes each direction's first-reachable Phase 6I seed and applies the Phase 6H-proven signed-theta
canonicalization before A03 actuation. Each independent reset runs 400×5 ms; per-target gates remain ≤15 mm
terminal error, ≥70% progress and no material violation, with aggregate strict 4/4.

Static validation PASSed 11/11 without runtime imports or environment steps. Phase 6F 0/4 remains immutable, and
this development gate cannot authorize VLA, broad workspace, final or hardware claims. Canonical config/static:
`configs/task/phase6j_feagine_a03_external_targets_recovery_dev4_v1.yaml`,
`results/phase6j_feagine_a03_external_targets_recovery_dev4_v1_static.yaml`.

Host runtime returned `REACHABILITY_CORRECTED_DEV4_FAIL`, 2/4. All targets passed IK, canonical nonnegative theta,
finite/exact dynamics, terminal error and progress. Final errors were 9.303–9.307 mm and progress
73.434–73.445%, so the reachability and representation corrections recovered target motion across all directions.

±Y fully PASSed. ±X failed only the material-limit gate: slide penetration was 0.263073/0.263067 mm, exceeding the
frozen 0.1 mm threshold. Maximum hinge penetration and contacts were zero. Internal/host wall was 0.459/0.87 s,
max RSS 382,640 KiB, zero swap/network; no prediction, reward or hardware command occurred. Preserve 2/4 FAIL.
Because v1 lacks the violating joint name and peak step, the next justified gate is an identical-target per-joint
limit diagnostic, not tolerance relaxation or target removal. Canonical result/manifest:
`results/phase6j_feagine_a03_external_targets_recovery_dev4_v1.yaml`,
`results/phase6j_feagine_a03_external_targets_recovery_dev4_v1_manifest.yaml`.

## 63. Phase 6K per-slide-joint diagnostic static PASS / host pending (latest; 2026-09-02)

Phase 6K repeats Phase 6J's four targets, frozen seeds, signed-theta mapping, controller, threshold and 400-step
budget exactly. It records every limited slide joint's name, range, extrema, first nominal violation step, peak
penetration step and magnitude, plus paired ±X/±Y counts. Static validation PASSed 10/10 with no runtime import or
environment step. Diagnostic completion cannot revise Phase 6J 2/4 or authorize any tolerance/controller change.
Canonical config/static: `configs/task/phase6k_feagine_a03_slide_limit_diagnostic_v1.yaml`,
`results/phase6k_feagine_a03_slide_limit_diagnostic_v1_static.yaml`.

Runtime diagnostic completed all four targets. +X's material violation was `feagine_grasper_joint_0`; -X's was the
mirrored `feagine_grasper_joint_1`. Both slide joints have range `[-0.013,0.005]` m, started at the fully-open lower
bound and reached approximately -0.01326307 m at the first simulation step. The opposite finger penetrated only
about 2.45 μm at step 33; ±Y penetrations were ≤3.90e-11 m. Thus the failure is a direction-dependent step-1
inertial transient at a hard-stop command, not a PCC proxy limit or sustained instability.

Internal/host wall was 0.413/0.61 s, max RSS 366,096 KiB, zero swap/network; no prediction, reward or hardware
command occurred. Phase 6J remains 2/4. A future gate may preregister small gripper-command margins away from the
hard stop while preserving all arm targets and the 0.1 mm material threshold. Canonical evidence:
`results/phase6k_feagine_a03_slide_limit_diagnostic_v1_manifest.yaml`, `scripts/verify_phase6k_assets.py`.

## 64. Phase 6L gripper hard-stop margin scan execution PASS / selection NO-GO (latest; 2026-09-02)

Phase 6L keeps all arm targets, seeds, representation, 400-step budget and gates unchanged. It reuses Phase 6J's
command-0.00 evidence and executes commands `[0.01,0.02,0.05]` for all four directions. The selection rule was
frozen as the smallest command achieving strict 4/4 with ≤15 mm error, ≥70% progress and ≤0.1 mm slide penetration.

Static validation PASSed 12/12. Host execution then completed 12 new trajectories/4,800 steps with exit 0. Commands
0.00, 0.01, 0.02 and 0.05 all remained strict 2/4. The mirrored ±X penetrations decreased monotonically from about
0.2631 to 0.2335, 0.2039 and 0.1150 mm; ±Y passed at every new command. All arm terminal errors and progress remained
about 9.30 mm and 73.44%, and all new state/time/IK checks passed. Command 0.05 nevertheless exceeded the frozen
0.1 mm threshold by about 0.015 mm, so `selected_gripper_command` is null and the decision is
`NO_SAFE_GRIPPER_MARGIN_SELECTED`.

Internal elapsed was 0.957 s for 24.0 s simulated time; host wall was 1.19 s, max RSS 448,968 KiB, zero swap/network,
CUDA/model/reward/hardware all absent. This is execution PASS but scientific selection FAIL/NO-GO. It cannot be
rounded into success or extended with a post-outcome value under Phase 6L; Phase 6J remains 2/4. Canonical evidence:
`results/phase6l_feagine_a03_gripper_margin_scan_v1_manifest.yaml`, `scripts/verify_phase6l_assets.py`.

## 65. Authorized recommended storage prune complete (latest; 2026-09-04)

The project occupied approximately 36 GiB, dominated by 31 GiB under `outputs/train`. With explicit user
authorization, a fixed allowlist removed 550 regenerable training-resume and noncanonical periodic-checkpoint files,
26.080067 GiB logical. The project is now 9.4 GiB; `/home` usage changed from 82% (30 GiB free) to 65% (56 GiB free).

Preserved: every structured result/raw log/presentation asset/failure directory; Phase 3B v2 protocol-collision and
invalid LIBERO duplicate evidence; final Diffusion EMA; final PushT deterministic/formal ACT; ACT-LIBERO
004460/049060/100000; canonical agentview and wrist QLoRA adapters. Removed payload paths, sizes and pre-delete
SHA-256 hashes are recorded in `results/storage_prune_recommended_20260904_manifest.yaml`; integrity is checked by
`scripts/verify_storage_prune_20260904.py`. Historical manifests that require deleted optimizer or periodic-checkpoint
bytes are now pre-prune archival evidence and cannot pass local presence checks. No experimental conclusion changed.
Post-prune documentation is frozen by `results/project_closeout_v16_storage_prune_manifest.yaml`.
