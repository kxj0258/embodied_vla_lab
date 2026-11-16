# Phase 4B: paired viewpoint development rollout

Date: 2026-08-28  
Status: `DEVELOPMENT_COMPLETE_DIRECT_SUBSTITUTION_NO_GO`

## Frozen question

How much closed-loop performance does the frozen single-image OpenVLA base retain when its third-person
`agentview` input is replaced directly by `eye_in_hand`, with task, init state, seed, instruction, action adapter,
checkpoint, preprocessing and horizon held fixed?

This measures direct viewpoint substitution compatibility. It does not claim learned viewpoint invariance and does
not add camera fusion, fine-tuning or an ego-view adapter.

## Development protocol

- Tasks `[0,3,6,7,8]`, init states `[0,1,2]`, views `[agentview, eye_in_hand]`: 15 paired states, 30 episodes.
- Each view gets a fresh environment explicitly bound to the same task/init and paired seed.
- Episode order is task, init, then agentview followed by eye-in-hand; horizon 600 and control frequency 20 Hz.
- One persistent frozen OpenVLA-base load at canonical revision `fa5ae1e750...9baf` serves all predictions.
- Both views receive the same 180-degree rotation and frozen OpenVLA 90% center crop.
- Primary reporting: strict success by view, Wilson intervals, paired outcome categories and task coverage.
- Step deltas are computed only where both views succeed. Timeout failures remain capped at 600 and disclosed.

The final task set `[1,2,4,5,9]` is untouched. Development may diagnose integration/preprocessing only and cannot
train or select a checkpoint. No failed episode may be removed, replaced or selectively rerun. Offline loss/token
accuracy/L1 are not environment success or reward and are not part of this evaluation.

Config: `configs/eval/phase4b_openvla_viewpoint_dev30_v1.yaml`  
Evaluator: `scripts/evaluate_phase4b_openvla_viewpoint_dev.py`  
Static checker: `scripts/check_phase4b_viewpoint_dev_static.py`

## Static result

The CUDA-free gate PASSed all 14 checks: exact development task metadata, view and init contracts, 30-episode/15-pair
arithmetic, canonical checkpoint revision, fresh targets, one persistent model load, fresh explicit-init environment
per episode, image-source-only view substitution, paired statistics, absence of training code, trajectory recording
and cleanup. It executed zero resets, steps and predictions. Canonical record:
`results/phase4b_openvla_viewpoint_dev30_v1_static.yaml`.

The earlier Phase 4A files contained a checkpoint revision metadata typo, recorded by
`results/phase4a_checkpoint_revision_metadata_correction.yaml`. Phase 4A loaded no model, so its camera result is
unaffected. Phase 4B binds the canonical revision from Phase 3A/P0 exactly.

## Development execution result

Execution PASSed all 30 episodes, but the direct eye-in-hand policy condition is a development NO-GO:

| View | Strict success | Wilson 95% | Timeouts | Task coverage | Reward sum |
|---|---:|---:|---:|---:|---:|
| agentview | 10/15 (66.67%) | 41.71%--84.82% | 5 | 5/5 | 10.0 |
| eye_in_hand | 0/15 (0%) | 0%--20.39% | 15 | 0/5 | 0.0 |

Paired outcomes were 0 both-success, 10 agentview-only, 0 eye-in-hand-only and 5 neither-success. The two-sided
exact paired binomial/McNemar value for the 10 versus 0 discordant pairs is 0.001953. This supports systematic
closed-loop degradation on this development set, not general viewpoint invariance or a final-set claim. Because
there were no both-success pairs, completion-step comparison is not applicable.

Agentview per-task counts for tasks `[0,3,6,7,8]` were `[2,2,1,3,2]/3`; eye-in-hand was `[0,0,0,0,0]/3`. All 15
wrist-view episodes reached the 600-step cap. Total environment steps/predictions were 13,663. Model load took 4.60
s; inference mean/median/P95 was 0.3401/0.3404/0.3483 s, 2.941 predictions/s. Peak allocated/reserved VRAM was
5,068,574,720/5,196,742,656 bytes. Internal/host wall was 4,962.71/4,966 s; host max RSS 5,867,244 KiB, zero
swap/network. All four shards loaded at the exact revision and worker shutdown PASSed with 13,663 predictions.

No training, loss, offline validation, token accuracy, L1 or gradient metrics apply. Final tasks `[1,2,4,5,9]`
remain untouched and are not authorized for the failed direct-substitution hypothesis. The next research step must
preregister a concrete ego-view adaptation hypothesis (for example representation alignment or an adapter) on
development data before any final evaluation; it cannot tune from the untouched final split.

The canonical CPU-only quantitative figure passed visual QA under
`results/phase4b_viewpoint_dev_summary_v2_manifest.yaml`; the overlapping-label v1 figure is preserved as
noncanonical. `scripts/verify_phase4b_assets.py` verifies the result, raw log, 13,663-line trajectory and canonical
PNG/SVG hashes without environment or model work.
