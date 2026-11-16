# Phase 5C: paired small-world-model training smoke

Date: 2026-08-29  
Status: `COMPLETE_ENGINEERING_PASS_OFFLINE_NO_GO`

## Frozen purpose

Test whether both Phase 5B models can complete a small, numerically stable optimization run and respond on held-out
windows. This is an engineering/offline smoke, not formal training or checkpoint selection.

The deterministic fixture contains references to 256 train and 128 validation H4 windows from the Phase 5A 40/10
episode split. It copies no images. Action-free and action-conditioned models each receive 100 updates, batch size 8,
the exact same batch order, and identical encoder/decoder initialization. Each is evaluated before and after on the
same 128 windows at horizons 1–4; exact copy-frame metrics are reported alongside both learned models.

The run must retain both step-100 checkpoints and optimizer states, finite losses/gradients, timing, throughput and
peak VRAM. No model is selected. Random improvement in a short smoke does not establish world-model quality,
action-conditioned superiority, reward prediction, planning utility or environment success. Formal training remains
forbidden regardless of the result until a separate protocol is frozen.

## Static result

All 14 checks PASSed: Phase 5A/5B prerequisites, fixture hash/counts, episode-safe H4 windows, disjoint split,
paired initialization/order, 100-update budget, H1–4 validation, no selection/formal-training authorization, fresh
targets and frozen trainer/model hashes. The checker initialized no CUDA and performed zero forward, optimizer or
environment work. Canonical result: `results/phase5c_world_model_train100_pair_v1_static.yaml`.

## Runtime result

Execution and checkpoint integrity PASSed. Both models completed 100 finite updates. First-ten to last-ten mean loss
fell 72.86% for action-free and 72.50% for action-conditioned; pre-clip gradient norms remained finite in ranges
0.0364–0.3222 and 0.0365–0.3193. Both step-100 checkpoints CPU-load with complete model and optimizer state.

The offline baseline decision is NO-GO. Post-smoke validation MAE is nearly flat around 11.49–11.53 for action-free
and 11.69–11.72 for action-conditioned across H1–4, while exact copy-frame is only 0.778/1.191/1.491/1.732. At H4,
conditioned is 1.71% worse than action-free and 576.96% worse than copy-frame. No horizon shows an action-conditioning
benefit. The short models learned coarse reconstruction, not competitive dynamics.

Total internal/host wall was 2.65/4.53 s; effective overall throughput was 75.47 updates/s. Peak allocated/reserved
VRAM was 141,130,240/165,675,008 bytes; host max RSS was 1,775,364 KiB, zero swap/network.

Decision: `ENGINEERING_PASS_OFFLINE_BASELINE_NO_GO`. No checkpoint is selected and formal training, planning,
candidate-action ranking or environment work is not authorized. Canonical result:
`results/phase5c_world_model_train100_pair_v1.yaml`.
