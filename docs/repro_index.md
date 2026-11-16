# Unified reproduction index

Updated: 2026-08-29

This index maps frozen claims to canonical evidence. Commands that train, run long GPU inference or execute MuJoCo
rollouts require explicit authorization and fresh IDs; they are not launched by this document. Existing historical
targets must never be overwritten.

## Read-only verification

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_p0_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase4b_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase4_closeout_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase5e_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase6a_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase6b_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase6c_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase6d_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase6e_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase6f_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase6g_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase6h_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase6i_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase6j_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase6k_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_phase6l_assets.py
/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_storage_prune_20260904.py
```

## Local storage profile

On 2026-09-04 the user authorized the recommended local-storage prune. Exactly 550 regenerable training payloads
(26.080067 GiB logical) were removed after recording their paths, byte sizes and SHA-256 values in
`results/storage_prune_recommended_20260904_manifest.yaml`. The project footprint fell from approximately 36 GiB
to 9.4 GiB. Results, raw logs, presentation media, failure directories, the Phase 3B v2 collision evidence and the
invalid LIBERO duplicate were preserved.

The retained inference assets are the final Diffusion EMA, final PushT deterministic/formal ACT models, ACT-LIBERO
checkpoints 004460/049060/100000, and the canonical agentview/wrist QLoRA adapters. Historical manifests still
identify deleted optimizer states and periodic checkpoints, but verifiers requiring those payload bytes are now
archival-pre-prune checks and are expected to fail local-file-presence checks. This does not revise any metric;
resuming the pruned training states requires regeneration or restoration from an external archive.

PushT run profiles may be inspected without execution:

```bash
EVLA_DRY_RUN=1 scripts/runs/pusht.sh diffusion-30k-train
```

## Evidence map

| Claim/stage | Canonical result | Config/protocol | Presentation or integrity entry |
|---|---|---|---|
| PushT ACT/Diffusion | `results/phase1b_act_pusht_deterministic_vs_vae_100seed.yaml`; `results/phase1c_diffusion_pusht_ema_030000_100seed.yaml` | `scripts/runs/pusht.sh`; `docs/pusht_reproducibility.md` | `results/phase1b_act_pusht_deterministic_30k_checkpoint_manifest.yaml`; `results/phase1b_act_pusht_formal_100k_checkpoint_manifest.yaml`; `results/phase1c_diffusion_pusht_baseline_30k_checkpoint_manifest.yaml` |
| ACT LIBERO 45/45 | `results/phase2b_act_libero_ckpt100000_final_heldout45.yaml` | Phase 2 configs and `scripts/evaluate_phase2b_act_libero.py` | `results/phase2b_act_libero_checkpoint_manifest.yaml` |
| OpenVLA base 45/45 | `results/phase3a_openvla_libero_final_heldout45_v1.yaml` | `configs/eval/openvla_libero_phase3a_final_heldout45.yaml` | `results/phase3a_openvla_checkpoint_manifest.yaml` |
| Agentview QLoRA 43/45 | `results/phase3b_openvla_qlora_step20_final_heldout45_v1.yaml` | Phase 3B configs/scripts | `results/phase3b_openvla_qlora_final_comparison.yaml` |
| P0 canonical assets | `results/p0_mvp_v2_manifest.yaml` | `docs/phases/p0_mvp_presentation.md` | `scripts/verify_p0_assets.py` |
| P0-HD comparisons | `results/p0_hd_comparisons_v1_manifest.yaml` | `docs/phases/p0_hd_demo_upgrade.md` | two 1920×1080 paired videos |
| Ten-task breadth 20/30 | `results/p0_breadth_openvla_base_10task30_v2.yaml` | `configs/eval/p0_breadth_openvla_base_10task30_v2.yaml` | `results/p0_breadth_10task_montage_v1_manifest.yaml` |
| Phase 4 camera gate | `results/phase4a_openvla_viewpoint_camera_gate_v2.yaml` | Phase 4A config/script | four native/direct RGB hashes |
| Direct wrist substitution | `results/phase4b_openvla_viewpoint_dev30_v1.yaml` | `configs/eval/phase4b_openvla_viewpoint_dev30_v1.yaml` | `results/phase4b_viewpoint_dev_summary_v2_manifest.yaml` |
| Wrist offline QLoRA | `results/phase4d_openvla_wrist_step20_vs_base_fullval889_paired.yaml` | Phase 4C/4D configs | `results/phase4d_wrist_fullval_summary_v1_manifest.yaml` |
| Wrist closed-loop NO-GO | `results/phase4e_openvla_wrist_adapter_task7_dev3_v1.yaml` | `configs/eval/phase4e_openvla_wrist_adapter_dev3_v1.yaml` | `results/phase4e_wrist_no_go_summary_v1_manifest.yaml` |
| Phase 4 closeout | `docs/phases/phase_4_viewpoint_closeout.md` | frozen A–E evidence chain | `results/phase4_viewpoint_closeout_v1_manifest.yaml`; `scripts/verify_phase4_closeout_assets.py` |
| World-model residual NO-GO | `results/phase5e_residual_world_model_dev100_pair_v1.yaml` | `configs/train/phase5e_residual_world_model_dev_smoke_v1.yaml`; `docs/phases/phase_5e_residual_world_model_dev_smoke.md` | `results/phase5e_residual_world_model_dev_summary_v1_manifest.yaml`; `scripts/verify_phase5e_assets.py` |
| Feagine bridge static | `results/phase6a_feagine_bridge_static_v4.yaml` | `configs/task/phase6a_feagine_bridge_static_v4.yaml`; `docs/phases/phase_6a_feagine_bridge_static.md` | `scripts/verify_phase6a_assets.py`; v1/v2 exceptions and v3 failure retained |
| Feagine A03 asset binding | `results/phase6b_feagine_a03_asset_contract_v1.yaml` | `configs/task/phase6b_feagine_a03_asset_contract_v1.yaml`; `docs/phases/phase_6b_feagine_a03_asset_contract.md` | `scripts/verify_phase6b_assets.py`; immutable wheel SHA-256 |
| Feagine A03 zero-step runtime | `results/phase6c_feagine_a03_runtime_zero_step_v2.yaml` | `configs/task/phase6c_feagine_a03_runtime_zero_step_v2.yaml`; `docs/phases/phase_6c_feagine_a03_runtime_zero_step.md` | `results/phase6c_feagine_a03_runtime_summary_v1_manifest.yaml`; `scripts/verify_phase6c_assets.py` |
| Feagine short dynamics + limit diagnosis | v1 `results/phase6d_feagine_a03_short_dynamics_v1.yaml` (6/7 FAIL); v2 `results/phase6d_feagine_a03_limit_diagnostic_v2.yaml` (PASS) | `configs/task/phase6d_feagine_a03_limit_diagnostic_v2.yaml`; `docs/phases/phase_6d_feagine_a03_short_dynamics.md` | `results/phase6d_feagine_a03_limit_diagnostic_v2_manifest.yaml`; v1 FAIL retained |
| Feagine bounded target reaching | `results/phase6e_feagine_a03_move_to_target_smoke_v1.yaml` (9/9 PASS) | `configs/task/phase6e_feagine_a03_move_to_target_smoke_v1.yaml`; `docs/phases/phase_6e_feagine_a03_move_to_target.md` | `results/phase6e_feagine_a03_move_to_target_smoke_v1_manifest.yaml`; `scripts/verify_phase6e_assets.py` |
| Feagine external target dev4 | `results/phase6f_feagine_a03_external_targets_dev4_v1.yaml` (0/4 FAIL) | `configs/task/phase6f_feagine_a03_external_targets_dev4_v1.yaml`; `docs/phases/phase_6f_feagine_a03_external_targets_dev.md` | `results/phase6f_feagine_a03_external_targets_dev4_v1_manifest.yaml`; `scripts/verify_phase6f_assets.py` |
| Feagine mapping diagnostic | `results/phase6g_feagine_a03_mapping_diagnostic_v1.yaml` (diagnostic PASS; Phase 6F remains 0/4) | `configs/task/phase6g_feagine_a03_mapping_diagnostic_v1.yaml`; `docs/phases/phase_6g_feagine_a03_mapping_diagnostic.md` | `results/phase6g_feagine_a03_mapping_diagnostic_v1_manifest.yaml`; `scripts/verify_phase6g_assets.py` |
| Feagine signed-theta/multi-start | `results/phase6h_feagine_signed_theta_multistart_static_v1.yaml` (4/5 FAIL) | `configs/task/phase6h_feagine_signed_theta_multistart_static_v1.yaml`; `docs/phases/phase_6h_feagine_signed_theta_multistart.md` | `results/phase6h_feagine_signed_theta_multistart_static_v1_manifest.yaml`; `scripts/verify_phase6h_assets.py` |
| Feagine reachability scan | v1 `results/phase6i_feagine_a03_reachability_scan_v1.yaml` (packaging FAIL); v2 recovery PASS | `configs/task/phase6i_feagine_a03_reachability_scan_v1.yaml`; `docs/phases/phase_6i_feagine_a03_reachability_scan.md` | `results/phase6i_feagine_a03_reachability_summary_v1_manifest.yaml`; `scripts/verify_phase6i_assets.py` |
| Feagine reachability-corrected dev4 | `results/phase6j_feagine_a03_external_targets_recovery_dev4_v1.yaml` (2/4 FAIL) | `configs/task/phase6j_feagine_a03_external_targets_recovery_dev4_v1.yaml`; `docs/phases/phase_6j_feagine_a03_external_targets_recovery.md` | `results/phase6j_feagine_a03_external_targets_recovery_dev4_v1_manifest.yaml`; `scripts/verify_phase6j_assets.py` |
| Feagine slide-limit diagnostic | `results/phase6k_feagine_a03_slide_limit_diagnostic_v1.yaml` (diagnostic PASS; Phase 6J remains 2/4) | `configs/task/phase6k_feagine_a03_slide_limit_diagnostic_v1.yaml`; `docs/phases/phase_6k_feagine_a03_slide_limit_diagnostic.md` | `results/phase6k_feagine_a03_slide_limit_diagnostic_v1_manifest.yaml`; `scripts/verify_phase6k_assets.py` |
| Feagine gripper margin scan | execution PASS; no safe command selected (all 2/4) | `configs/task/phase6l_feagine_a03_gripper_margin_scan_v1.yaml`; `docs/phases/phase_6l_feagine_a03_gripper_margin_scan.md` | `results/phase6l_feagine_a03_gripper_margin_scan_v1_manifest.yaml`; `scripts/verify_phase6l_assets.py` |

## Evidence semantics

- `loss`, token accuracy and decoded L1 are teacher-forced offline metrics.
- `reward` and strict success come only from environment execution.
- Development states may diagnose or select only according to rules frozen before their outcomes.
- Held-out final evidence is one-shot and cannot be used to retune or replace failures.
- Presentation recaptures and offline video composition cannot revise original success estimates.
- A manifest proves local artifact identity, not that ignored large files are included in a future public clone.

## Fresh reproduction rule

A new execution must use a new experiment ID, output directory, raw log, structured result and manifest entry. Before
running, verify pinned code/model/data identities and resource gates. After running, classify execution separately
from policy outcome and record all failures, timing, memory, checkpoint integrity and cleanup evidence.
