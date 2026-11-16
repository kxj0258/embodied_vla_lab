# Embodied VLA Lab — project report

Updated: 2026-08-29  
Project status: `P0_PRESENTATION_COMPLETE_PHASE4_COMPLETE_NO_GO_PHASE5_DEVELOPMENT_NO_GO`

## Executive summary

Embodied VLA Lab is a simulation-first robotics learning project built around LeRobot, LIBERO and OpenVLA. It
implements reproducible data/action contracts, imitation-learning baselines, language-conditioned manipulation,
4-bit VLA inference, QLoRA adaptation, headless HD demonstration capture and preregistered viewpoint experiments on
one 12 GB RTX 5070 Ti Laptop GPU.

The strongest reliable manipulation results are ACT and frozen OpenVLA base at 45/45 strict success on the complete
held-out finite set of one LIBERO-Goal task. A separate all-ten-task portfolio protocol achieved 20/30 strict success
and at least one success on every task. QLoRA repeatedly improved teacher-forced metrics but did not improve policy
reliability: agentview final fell from 45/45 to 43/45, while wrist-view adaptation improved offline metrics but
remained 0/3 in its preregistered closed-loop recovery gate. The project therefore keeps frozen base as the reliable
demo policy and treats both adapters as research assets.

## System design

The project separates the LIBERO/MuJoCo process from the OpenVLA process because their dependency stacks differ.
Images, instructions and actions cross a schema-validated PNG/JSON IPC boundary with request/response hashes,
checkpoint revision checks, action bounds, timeouts and deterministic worker cleanup. Every substantive experiment
uses immutable IDs, refuses overwrite, saves structured results and preserves failures.

![System architecture](../outputs/presentation/p0_mvp_v2_20260827/embodied_vla_lab_architecture.png)

## Experimental results

| Study | Protocol | Frozen result | Interpretation |
|---|---|---|---|
| PushT baselines | Same dataset revision; independent 100-seed tests | deterministic ACT 6%, VAE ACT 2%, Diffusion EMA 4% strict success | Diffusion improved dense reward but was about 39× slower than deterministic ACT and did not improve sparse success |
| ACT on LIBERO | checkpoint selection on 5 dev states; remaining states 5–49 as final | 45/45 strict success | Complete enumeration of the frozen finite set, not cross-task generalization |
| OpenVLA frozen base | same held-out states 5–49 | 45/45; median/P95 inference about 0.409/0.433 s | Reliable single-task demo policy |
| Agentview rank-8 QLoRA | paired fullval889, dev5, then one-shot held-out final | offline loss/token accuracy/L1 improved; dev 5/5; final 43/45 | Offline adaptation PASS, environment non-regression FAIL |
| Ten-task breadth | all LIBERO-Goal task IDs × 3 fixed init states | 20/30; 10/10 tasks covered; ten timeouts retained | Small portfolio breadth set, not official full benchmark |
| Direct wrist substitution | 5 dev tasks × 3 states × paired agent/wrist views | agent 10/15 vs wrist 0/15; exact paired p=0.001953 | Systematic development degradation; final split untouched |
| Wrist rank-8 QLoRA | task-7 wrist demos; paired fullval889 | loss −45.77%, token accuracy +7.73 pp, continuous L1 −32.43%; 10/10 episodes improve | Strong teacher-forced adaptation |
| Wrist closed-loop recovery | task 7, init 0–2, same seeds as immutable base reference | base 0/3, adapter 0/3; all 600-step timeouts, reward 0 | Closed-loop recovery NO-GO |
| Small world model | real LIBERO frames; paired 100-update action-free/conditioned development smokes | full-frame H4 11.526/11.722 vs copy 1.732; residual H4 1.59223/1.59303 vs copy 1.59125 | Engineering paths PASS, learned dynamics/action conditioning NO-GO |
| Feagine bridge static | provisional single-section PCC; 256 deterministic round trips and fail-closed cases | 14/14 PASS; max FK/IK round-trip error 8.58e-16 m | Software contract only; A03 assets and simulation are absent |
| Feagine A03 asset binding | v1.0.1 distribution archive and A03 Type-2 preset | 19/19 static PASS; 3×0.16 m sections, 48 proxy joint/actuator pairs, 5 ms timestep | Actual archive contract established; runtime not installed or loaded |
| Feagine A03 runtime/dynamics | isolated core build + local wheel, explicit A03 Type-2 | corrected targets recover motion in 4/4; strict 2/4 traced to mirrored grasper fingers transiently crossing the open hard stop by 0.263 mm at step 1 | Margin scan reduced penetration monotonically to 0.115 mm but did not pass the frozen 0.1 mm gate; no VLA or workspace-generalization claim |
| Feagine short dynamics | 200 headless steps, one simulated second | 6/7 FAIL: finite state and 6.70 cm tip motion, but joint-limit check failed | Per-joint penetration diagnostic required before any target-reaching gate |

## What the negative results establish

The two QLoRA studies show why offline action prediction cannot substitute for environment evaluation. On agentview,
offline improvement and a perfect five-state development result still preceded two held-out failures. On wrist view,
the adapter improved every validation episode on all three offline metrics yet recovered zero closed-loop states.
These outcomes justify the project's protocol rules: checkpoint selection must precede final evaluation; final
failures cannot trigger retuning; loss, token accuracy and L1 are never reported as success or reward.

![Viewpoint evidence chain](../outputs/presentation/phase4_viewpoint_closeout_v1_20260828/phase4_viewpoint_evidence_chain.png)

## Demonstration assets

- Canonical ten-task OpenVLA montage: 1920×1080, 20 fps, 64.8 s, ten genuine 640-rendered success recaptures.
- ACT and frozen-base OpenVLA single-task success captures.
- Base-versus-QLoRA paired HD examples: one typical shorter control trajectory and one tail failure.
- Quantitative figures for reliability, offline adaptation, task breadth and viewpoint robustness.

The ten montage clips are selected by a frozen first-success rule and do not replace the disclosed 20/30 aggregate.
Presentation frame rate is not policy control frequency or model throughput.

## Reproducibility

Git HEAD and all three upstream submodule pins are frozen in handoff evidence. Checkpoint/model revisions, local
asset hashes, split definitions, init states, seeds, raw logs, trajectories and media hashes are recorded. Static
gates run before GPU/environment work, and all new runs use unique targets with overwrite refusal.

Use [the unified reproduction index](repro_index.md) to locate each canonical config, result, manifest, verifier and
host-execution boundary. Existing outputs intentionally prevent historical commands from being rerun in place.

The authorized 2026-09-04 storage prune removed 26.080 GiB of optimizer state and noncanonical periodic checkpoint
payloads while retaining canonical inference models, results, raw logs, failures and presentation media. A
pre-deletion SHA-256 inventory preserves artifact identity, but deleted training states are no longer locally
resumable; old checkpoint-presence verifiers must be interpreted as pre-prune archival checks.

## Limitations

- The 45/45 results are complete enumerations of one task's frozen finite init-state set, not a universal success rate.
- The ten-task protocol uses only three states per task and is not the official full LIBERO benchmark.
- Viewpoint evidence is development-only and concludes NO-GO; it does not establish ego-view robustness.
- No real-robot, simulator-transfer or language-generalization result is claimed.
- OpenVLA inference is slower than a 20 Hz control period; video playback rates are presentation choices.
- The small-world-model studies are short offline development smokes; neither learned model beat copy-frame, so no planning, reward, success or environment utility is claimed.
- The A03 archive contract and bounded dynamics are verified. Reachability and signed-angle corrections recovered motion for four corrected Cartesian targets, but the strict safety outcome remains 2/4; a preregistered gripper-margin scan reduced the mirrored hard-stop transient to 0.115 mm without passing the 0.1 mm gate. No VLA-transfer or hardware result is claimed.

## Portfolio conclusion

The project demonstrates end-to-end embodied-AI engineering under tight compute constraints: data contracts,
training, quantized VLA inference, dependency isolation, rigorous environment evaluation, reproducible media and
honest negative-result analysis. The most defensible message is not that every adaptation worked, but that the
system reliably distinguishes offline fitting, development behavior, final reliability and deployment evidence.
