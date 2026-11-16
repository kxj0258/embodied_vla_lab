# Phase 6L — gripper hard-stop margin scan

Phase 6L preserves all Phase 6J arm targets, IK seeds, representation, 400-step budget and safety thresholds. It
reuses command 0.00 evidence and executes commands 0.01, 0.02 and 0.05. The preregistered selection is the smallest
command achieving strict 4/4 while retaining ≤15 mm error, ≥70% progress and ≤0.1 mm slide penetration.

This is a development safety-margin scan, not threshold relaxation. Phase 6J remains 2/4 regardless of outcome;
no VLA, broad workspace, final evaluation, rendering or hardware is included.

## Runtime result

Host execution completed normally (`HOST_EXIT_STATUS=0`): 12 new trajectories, 4,800 MuJoCo steps and 24.0 s
simulated time completed in 0.957 s internal elapsed time. Peak host RSS was 448,968 KiB; CUDA was not initialized,
there were no model predictions, rewards, hardware commands, swaps or network traffic.

The execution classification is **PASS**, but the preregistered selection outcome is
`NO_SAFE_GRIPPER_MARGIN_SELECTED`:

| gripper command | strict successes | +X penetration | -X penetration |
|---:|---:|---:|---:|
| 0.00 (reused) | 2/4 | 0.263073 mm | 0.263067 mm |
| 0.01 | 2/4 | 0.233468 mm | 0.233461 mm |
| 0.02 | 2/4 | 0.203863 mm | 0.203856 mm |
| 0.05 | 2/4 | 0.115047 mm | 0.115040 mm |

All ±Y episodes passed with zero material penetration for the new commands. Every new episode retained finite state,
exact time, successful IK, about 9.30 mm terminal error and 73.44% progress. Command 0.05 therefore missed the frozen
0.1 mm safety threshold by about 0.015 mm on both ±X targets. No command is selected; Phase 6J remains 2/4. The
monotone trend is mechanism evidence only and cannot authorize an unregistered post-outcome command or revise this
NO-GO.
