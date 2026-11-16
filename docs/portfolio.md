# Embodied VLA Lab — portfolio brief

## What I built

A simulation-first embodied-learning lab spanning reproducible robot-data contracts, ACT and Diffusion Policy
training, language-conditioned LIBERO control, quantized OpenVLA inference and parameter-efficient adaptation. The
system runs on one 12 GB RTX 5070 Ti Laptop GPU and separates the LIBERO/MuJoCo runtime from the OpenVLA runtime via
hash-bound PNG/JSON IPC.

![System architecture](../outputs/presentation/p0_mvp_v2_20260827/embodied_vla_lab_architecture.png)

## Results that matter

- ACT-on-LIBERO: trained deterministic ACT for 100k steps; the checkpoint selected on five development states
  achieved 45/45 strict success on every remaining frozen init state. A separately identified presentation capture
  reproduced held-out init5 success in 83 steps without changing the final protocol.
- OpenVLA-7B: ran the official LIBERO-Goal checkpoint in NF4/BF16; frozen base also achieved 45/45 on the same
  held-out finite set with median/P95 inference of about 0.409/0.433 seconds.
- Multi-task breadth: preregistered all ten LIBERO-Goal task IDs with three fixed initial states each. Frozen base
  achieved 20/30 strict success and 10/10 task coverage; a fixed-selection 10/10 genuine-640 success recapture set
  and 1920x1080 64.8 s montage are complete. The selected demos do not replace the disclosed ten timeout failures.
  This is a small catalog-wide portfolio set, not the official full benchmark.
- Rank-8 QLoRA: adapted only 0.7048% of parameters. On the paired 889-frame validation set, loss decreased 47.36%,
  action-token accuracy increased 10.33 percentage points and continuous-action L1 decreased 32.63%.
- Closed-loop reality check: QLoRA improved typical successful-state efficiency but failed init12 and init44,
  reducing final reliability from 45/45 to 43/45. I retained the base model for reliable demonstration rather than
  selecting solely by teacher-forced offline loss.
- Ego-view ablation: validated native wrist-camera rendering, then preregistered paired development tests. Direct
  substitution changed frozen-base success from 10/15 agentview to 0/15 wrist. A task-specific rank-8 wrist QLoRA
  improved paired full-validation loss by 45.77%, token accuracy by 7.73 percentage points and continuous L1 by
  32.43%, yet remained 0/3 in task-7 closed loop with three 600-step timeouts. This isolates a measured
  teacher-forced-to-control generalization gap rather than claiming successful viewpoint transfer.

![Held-out reliability](../outputs/presentation/p0_mvp_v2_20260827/heldout_reliability.png)

![Phase 4 viewpoint evidence chain](../outputs/presentation/phase4_viewpoint_closeout_v1_20260828/phase4_viewpoint_evidence_chain.png)

## Engineering highlights

- Immutable experiment IDs, strict no-overwrite guards, structured results and SHA-256 manifests.
- Exact checkpoint identity and CPU reload checks; preserved failed gates and protocol collisions as evidence.
- Dependency-isolated OpenVLA bridge with request/response schema validation, hashes, timeouts, action bounds,
  gripper adaptation and deterministic cleanup.
- Explicit separation of training loss, offline validation, environment reward and strict success.
- Produced genuine-640 simulator captures with native-128 frozen policy input, plus 1920x1080 paired videos showing
  both a typical QLoRA control-path speedup (70 versus 221 steps) and a reliability-tail regression (600-step QLoRA
  timeout versus 68-step base success). QLoRA inference itself was slower, so the comparison does not conflate
  shorter control trajectories with faster model execution.

## Scope

The strongest reliability results cover one task, one simulator/runtime and fixed finite init-state sets. The
10-task breadth set establishes limited catalog coverage, while Phase 4 establishes a development-set viewpoint
NO-GO—not viewpoint generalization. Nothing establishes language, simulator or real-world transfer.

P0-HD media: `outputs/presentation/p0_hd_comparisons_v1_20260828`; hashed record:
`results/p0_hd_comparisons_v1_manifest.yaml`.

Phase 4 closeout: `docs/phases/phase_4_viewpoint_closeout.md`; hashed visual record:
`results/phase4_viewpoint_closeout_v1_manifest.yaml`.
