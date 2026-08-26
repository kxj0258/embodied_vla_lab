# PushT reproducibility and command asset audit

Updated: 2026-08-26.

## Audit conclusion

The project had strong configuration and result provenance before this audit, but incomplete executable command
provenance. Training YAMLs, offline checkers, structured YAML results, checkpoint hashes, and artifact hashes were
tracked. Most host training and rollout commands were recoverable only from `Command being timed` inside
`results/raw/*.log`, which is intentionally ignored by Git; a subset of ACT commands was copied into a README or
handoff. Therefore a fresh clone had enough information to reconstruct the runs, but did not have one safe,
versioned entrypoint for every critical run.

This gap is now closed by `scripts/runs/pusht.sh`. It contains named, immutable profiles for the completed ACT and
Diffusion Policy preflights, training stages, checkpoint audits, fixed five-seed selection, paired 100-seed tests,
ACT replanning/temporal-ensemble diagnostics, and Diffusion demo rendering. Profiles enforce strict offline flags, the `evla-core` environment, fixed paths and
seeds, `/usr/bin/time -v`, log hashing, and refusal to overwrite existing outputs.

## What was already saved

| Asset | Git tracked before audit | Role |
|---|---:|---|
| `configs/dataset/pusht_v3.yaml` | yes | dataset revision and 185/21 split |
| `configs/train/*.yaml` | yes | all completed ACT and Diffusion training contracts |
| `scripts/check_phase*.py` | yes | dataset/config/checkpoint offline gates |
| `results/*.yaml` | yes | metrics, statistical boundaries, paths, and SHA-256 manifests |
| experiment READMEs and handoffs | yes | narrative decisions and many commands |
| `results/raw/*.log` | no, ignored | full stdout/stderr and exact timed command on the current host |
| `outputs/` | no, ignored | checkpoints, eval JSON, and videos on the current host |
| standalone host run scripts | incomplete | added by this audit |

Generated artifacts remain intentionally outside Git because the Diffusion checkpoint tree alone is about 15 GB.
Their identities are preserved by tracked manifests. Reproduction still requires the pinned dataset snapshot and,
for exact checkpoint identity, rerunning on a compatible CUDA/software stack. Diffusion trajectories are stochastic
and were empirically not bitwise stable across processes even with the same nominal seeds.

## Safe use

List profiles without running anything:

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab
conda activate evla-core
scripts/runs/pusht.sh --list
```

Resolve and inspect one command without executing it:

```bash
EVLA_DRY_RUN=1 scripts/runs/pusht.sh diffusion-30k-train
```

Run only after the corresponding installation, training, or evaluation action has been explicitly authorized:

```bash
scripts/runs/pusht.sh diffusion-ema-30k-5ep
```

Existing outputs intentionally make historical profiles fail with exit code 3. This prevents an accidental rerun
from overwriting evidence. To reproduce in a fresh workspace, preserve the same submodule pins, dataset revision,
environment package versions, and filesystem paths, or deliberately update the tracked contracts first.

## Completed experiment map

| Stage | Reproduction profile | Structured result |
|---|---|---|
| ACT smoke | `act-smoke-train` | `results/phase1b_act_pusht_smoke_2step.yaml` |
| ACT single-episode overfit | `act-overfit-train` | `results/phase1b_act_pusht_overfit_ep51_2000step.yaml` |
| ACT batch-8 resource gate | `act-resource-gate-train` | `results/phase1b_act_pusht_fullsplit_batch8_100step.yaml` |
| ACT VAE 100k | `act-formal-100k-train` | `results/phase1b_act_pusht_formal_100k.yaml` |
| Deterministic ACT 30k | `act-deterministic-30k-train` | `results/phase1b_act_pusht_deterministic_30k.yaml` |
| ACT held-out comparison | `act-deterministic-30k-100seed`, `act-vae-25k-100seed` | `results/phase1b_act_pusht_deterministic_vs_vae_100seed.yaml` |
| Diffusion resource gate | `diffusion-resource-gate-train` | `results/phase1c_diffusion_pusht_resource_gate_2step.yaml` |
| Diffusion 30k | `diffusion-30k-train` | `results/phase1c_diffusion_pusht_baseline_30k.yaml` |
| Diffusion checkpoint selection | `diffusion-ema-{10k,20k,30k}-5ep` | `results/phase1c_diffusion_pusht_checkpoint_selection_5ep.yaml` |
| Diffusion held-out test | `diffusion-ema-30k-100seed` | `results/phase1c_diffusion_pusht_ema_030000_100seed.yaml` |
| Diffusion success videos | `diffusion-demo-replay31` | same result plus checkpoint manifest |

## Current measured outcome

- Deterministic ACT 30k: 6/100 held-out successes, mean max reward 0.51557, about 1.17 seconds per episode.
- VAE ACT 25k: 2/100 held-out successes, mean max reward 0.41226.
- Diffusion EMA 30k: 4/100 held-out successes, mean max reward 0.65133, about 45.55 seconds per episode.
- Diffusion improves paired dense reward and peak proximity versus both ACT baselines, but does not establish a
  stable sparse-success improvement over deterministic ACT.
- Two strict-success Diffusion videos exist from the 31-episode sequential rerun; exact individual trajectory
replay failed, so they are demo samples rather than evidence of deterministic replay.

The Phase 1C environment additionally requires `diffusers==0.39.0`, matching the pinned LeRobot lockfile constraint
`>=0.38.0,<0.40.0`. Installing it is intentionally not automated by the run profile because installation remains a
separate authorization and network action.

No profile authorizes itself. Downloads, installations, training, evaluation, upload, commit, and push remain
separate user decisions.
