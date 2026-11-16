# P0-Breadth OpenVLA multi-task demonstration

Date: 2026-08-28  
Status: `COMPLETE`

## Purpose

Expand the portfolio beyond the already-complete single-task ACT/OpenVLA evidence without new training or
post-outcome task selection. The reliable frozen OpenVLA base is evaluated on all ten tasks in the LIBERO-Goal task
catalog. This is still not the official full benchmark and does not revise the Phase 3A `turn_on_the_stove` 45/45
result. The earlier five-task v1 static plan was never executed and is preserved as superseded history.

## Frozen protocol

- Suite/checkpoint: LIBERO-Goal; OpenVLA-7B frozen base revision `fa5ae1e...baf` in NF4/BF16.
- All ten catalog tasks fixed before execution: task IDs 0--9, including articulation, pushing, container/rack/stove
  placement and a compound drawer-and-bowl task.
- Init states `[0,1,2]` per task; 30 episodes total; 600-step cap; seeds 20261001--20261030.
- Native 128x128 agentview input, frozen 180-degree rotation/OpenVLA center-crop preprocessing, 20 Hz environment
  contract, one persistent model load and dependency-isolated hashed IPC.
- Every task/state remains in the aggregate after outcomes are observed. Failures cannot be removed or replaced.

## Claims boundary

Report episode success as `x/30`, per-task success as `x/3`, task coverage and failure modes. Do not call this
general LIBERO performance, language generalization or a statistically independent final benchmark. Offline loss,
token accuracy and L1 are not environment success or reward. Representative 640 HD recaptures may be made only
after the fixed evaluation, under new presentation-only IDs, and cannot change the aggregate.

Config: `configs/eval/p0_breadth_openvla_base_10task30_v2.yaml`  
Evaluator: `scripts/evaluate_p0_breadth_openvla_base.py`

## Frozen execution result

Execution PASSed for all 30 preregistered episodes. Frozen base achieved 20/30 strict success (66.67%; Wilson 95%
48.78%--80.77%) and at least one success on every catalog task (10/10 task coverage). Per-task results were:

| ID | Short task | Success | Steps by init 0/1/2 |
|---:|---|---:|---|
| 0 | open middle drawer | 2/3 | 114 / 600F / 123 |
| 1 | bowl on stove | 2/3 | 600F / 90 / 73 |
| 2 | wine bottle on cabinet | 3/3 | 77 / 99 / 333 |
| 3 | open drawer + bowl inside | 1/3 | 600F / 600F / 246 |
| 4 | bowl on cabinet | 3/3 | 93 / 93 / 103 |
| 5 | push plate to stove front | 2/3 | 133 / 600F / 135 |
| 6 | cream cheese in bowl | 1/3 | 600F / 108 / 600F |
| 7 | turn on stove | 3/3 | 89 / 81 / 84 |
| 8 | bowl on plate | 1/3 | 76 / 600F / 600F |
| 9 | wine bottle on rack | 2/3 | 270 / 134 / 600F |

There were 10 timeout failures. Successful episodes used mean/median 127.7/101 steps; capped all-episode
mean/median was 285.13/133.5. Total environment steps/predictions were 8,554. Model inference
mean/median/P95 was 0.3346/0.3331/0.3491 s (2.989 predictions/s). Peak allocated/reserved VRAM was
5,068,574,720/5,196,742,656 bytes. Host wall was 3,066.95 s, maximum RSS 5,773,664 KiB, with zero swap/network and
clean worker/GPU shutdown. Frozen checkpoint revision matched.

Training/offline validation/token accuracy/L1/gradients are not applicable. The valid claim is catalog-wide task
coverage on this fixed 30-episode set, not general LIBERO performance. Representative successful HD recaptures may
now use the already-recorded first-success state for each task, but selection must be explicit and cannot alter 20/30.

## Presentation assets and HD selection

The CPU-only summary figure/JSON are PASS under `results/p0_breadth_summary_v1_manifest.yaml`; generation used zero
environment steps and predictions. For HD demos, the selection rule is frozen as the first successful init in task
and init order: `[(T0,0),(T1,1),(T2,0),(T3,2),(T4,0),(T5,0),(T6,1),(T7,0),(T8,0),(T9,0)]`. Their previously
observed steps are `[114,90,77,246,93,133,108,89,76,270]`.

`scripts/record_p0_breadth_hd_demos.py` is static PASS with 21 fresh targets. It reuses native 128 policy input plus
same-sim/single-EGL direct 640 display, and asserts every recapture matches its frozen successful step count. It does
not rerun failures or modify 20/30.

## HD execution and presentation completion

The fixed ten-capture batch PASSed 10/10 with exact frozen step counts, totalling 1,296 environment steps and model
predictions. Host exit status was 0; wall/user/system time was 785.02/1089.10/110.11 s, CPU utilization 152%, and
maximum RSS 5,860,536 KiB, with zero swap and network traffic. Per-capture median inference latency was
0.3315--0.3388 s and P95 was 0.3363--0.3457 s. Peak allocated VRAM was at most 5,068,574,720 bytes and reserved VRAM
was 5,196,742,656 bytes. All checkpoint revisions matched, workers shut down cleanly, and the GPU compute process was
absent after the run.

All twenty 640-source presentation/clean masters passed codec, frame-count and hash checks. The original Task 6
presentation encode is preserved but marked noncanonical after visible early-frame block corruption; its clean master
was intact and was recomposed offline with zero environment steps or model predictions. Multi-timepoint visual QA
passed for the repaired Task 6 video and the 10-task montage.

Canonical 1920x1080, 20 fps, 64.8 s portfolio montage:
`outputs/presentation/p0_breadth_10task_montage_v1_20260828/openvla_10task_success_montage_hd.mp4`.
Structured records: `results/p0_breadth_hd_10task_v1_manifest.yaml` and
`results/p0_breadth_10task_montage_v1_manifest.yaml`. P0-Breadth is COMPLETE; these selected success demos illustrate
task coverage, while the frozen reliability claim remains 20/30 and all ten timeout failures remain disclosed.
