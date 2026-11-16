# Phase 4E: task-7 wrist-adapter development rollout

Date: 2026-08-28  
Status: `COMPLETE_NO_GO`

## Hypothesis and evidence boundary

The rank-8 step-20 wrist adapter may recover closed-loop control on its training task after Phase 4B showed that
direct frozen-base wrist substitution failed. This is a development-only hypothesis, not cross-task or final
generalization. Offline Phase 4D improvement is motivation only and is not success or reward.

Phase 4B already provides immutable frozen-base wrist evidence on task 7, init states `[0,1,2]`, seeds
`[20261120,20261121,20261122]`: 0/3 success and three 600-step timeouts. Phase 4E therefore does not rerun base. It
runs the fixed step-20 adapter once on each exact same state/seed, keeps every success/failure/timeout, and reports
strict success, reward, completion steps, inference latency, resources and artifact hashes.

## Preregistered decision rule

- 0/3 adapter successes: `NO_GO_NO_CLOSED_LOOP_RECOVERY`.
- 1/3 or 2/3: `DEVELOPMENT_RECOVERY_SIGNAL_PARTIAL`.
- 3/3: `DEVELOPMENT_RECOVERY_SIGNAL_STRONG`.

No outcome automatically authorizes final tasks, checkpoint tuning/selection, failure replacement or a claim of
general ego-view robustness. The adapter was trained only on task 7; final tasks `[1,2,4,5,9]` remain untouched.

## Static result

All 12 checks PASSed: canonical revision and adapter hash, task-7-only scope, exact three Phase 4B states/seeds/base
rows, wrist-only observation, frozen evaluator/worker hashes, decision rule, fresh targets and final prohibition.
The checker initialized no CUDA and performed zero reset, step or prediction. Canonical result:
`results/phase4e_openvla_wrist_adapter_task7_dev3_v1_static.yaml`.

## Runtime result and conclusion

Execution PASSed, but the preregistered policy decision is `NO_GO_NO_CLOSED_LOOP_RECOVERY`. The adapter obtained
0/3 strict success; all three init states ran the full 600 steps with sum/max reward 0 and no termination. Against
the immutable Phase 4B base reference, all three paired outcomes are `neither_success`. Thus the strong Phase 4D
teacher-forced improvements did not recover task-7 wrist closed-loop control.

The run completed 1,800 finite in-range predictions. Model load was 6.62 s; mean/median/P95 inference latency was
0.4876/0.4880/0.5033 s, 43.02% slower in mean latency than the matching Phase 4B base wrist episodes. Peak
allocated/reserved VRAM was 5,468,073,984/5,584,715,776 bytes. Internal/host wall was 931.18/937.42 s; host max RSS
was 5,774,904 KiB with zero swap/network. Worker shutdown at exactly 1,800 predictions and artifact hashes PASSed.

No loss, token accuracy or gradient metrics apply to this rollout. No training/checkpoint selection occurred. This
hypothesis is closed: do not retune, switch checkpoints, selectively rerun these states or execute final tasks from
this evidence. Canonical result: `results/phase4e_openvla_wrist_adapter_task7_dev3_v1.yaml`.
