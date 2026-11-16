# Phase 3B — OpenVLA QLoRA feasibility

Updated: 2026-08-27  
Current gate: source/data contract PASS; minimal PEFT runtime addition pending.

Phase 3A froze a strong official base-policy baseline before any project training: 5/5 development and 45/45
held-out strict success on `turn_on_the_stove`. QLoRA is therefore a resource/engineering and controlled adaptation
study, not a repair for a policy that cannot solve the base task. The 593-step held-out tail remains a possible
efficiency target, but final states cannot be used for training or hyperparameter selection.

The first resource gate uses the frozen NF4 checkpoint, BF16 compute, `prepare_model_for_kbit_training`, rank-8 LoRA
with `target_modules="all-linear"`, batch size 1, and exactly one forward/backward without an optimizer step. Rank 8
is intentionally below upstream rank 32 to establish the 12 GB feasibility floor. A later gate may increase rank only
under a new ID.

The gate will not install or import the upstream RLDS/TensorFlow/dlimp stack. A single real sample is exported from
the verified local LIBERO HDF5. Its raw environment gripper action is transformed exactly as upstream LIBERO training:
`1 - clip(raw_gripper, 0, 1)`. `ActionTokenizer` produces seven tokens; labels outside action tokens are `-100` and
the initial resource gate ignores the stop token. Metrics must include loss, action-token accuracy, continuous-action
L1, trainable parameters, gradient norm, allocated/reserved VRAM, RSS and forward/backward time.

Current `evla-openvla` inventory has no PEFT, TRL or HF datasets package. Only upstream-pinned `peft==0.11.1` is
needed; install it with `--no-deps` to prevent mutation of the frozen Torch/Transformers/Accelerate stack. The import
checker initializes no CUDA and loads no model. No QLoRA forward/backward or training has run.

The PEFT runtime gate is now PASS. Exact PEFT 0.11.1 was installed with `--no-deps`; Torch 2.11.0+cu128,
Transformers 4.40.1 and Accelerate 1.10.1 remained unchanged, and `pip check` reports no broken requirements. The
import checker initialized no CUDA and loaded no model.

The real-sample resource fixture is also frozen. It uses `demo_0/t=0`, source HDF5 SHA-256 `387fc107...aac73d5`,
the already frozen image SHA-256 `d4c985...78dbd`, and raw action
`[0.286607,0.136607,0.002679,0,0.128571,-0.066429,-1]`. Upstream LIBERO transformation makes its training gripper
component +1. The one-forward/backward entrypoint and no-CUDA static checker PASS. Formal GPU execution remains
pending and will not take an optimizer step or write a checkpoint.

The first formal v1 resource attempt is FAIL after compute. All four shards loaded; the script passed its finite-loss,
action-logit alignment, backward and finite-gradient assertions and constructed the result dictionary. It then failed
because the path variable `output` had been overwritten by the model forward output, so `.mkdir()` was called on a
`PrismaticCausalLMOutputWithPast`. This is not OOM or an observed numerical failure, but loss/accuracy/L1/gradient/
VRAM values were not serialized and cannot be reconstructed. Process wall was 9.34 s and max RSS 5,775,856 KiB.

The v1 log and source/config hashes are frozen. V2 changes only the local names to `output_dir` and `model_output`,
uses a new ID/output/log, and passes both the no-CUDA writer regression and formal GPU gate. The one real sample gave
loss 17.8423, 0/7 action-token accuracy, continuous-action L1 0.4807 and finite gradient norm 539.24. These are an
untrained one-sample baseline, not offline validation or environment performance. Forward/backward took 0.622/0.463 s;
peak allocated/reserved VRAM was 6,621,465,600/6,851,395,584 bytes, host wall 8.44 s and max RSS 5,835,192 KiB.
There was no optimizer step or checkpoint. The next resource gate must allocate optimizer state, clip the large
single-sample gradient and perform exactly one parameter update under a new experiment ID before formal training.

Optimizer gate v1 is FAIL at its post-step assertion, not OOM. Forward, backward and the `optimizer.step()` call were
reached, but the verifier inspected only the first trainable LoRA parameter. With Gaussian LoRA initialization a
specific first-step parameter can legitimately have zero gradient/update, so that sentinel cannot represent the
whole adapter. No result was serialized; numerical metrics and optimizer peak VRAM are therefore unavailable. V2
must select a parameter with a proven finite nonzero pre-step gradient before checking its post-step delta.

Optimizer gate v2 is PASS. The pre-clip norm was 539.247 and clipping to 1.0 completed with finite gradients. AdamW
allocated 218,256,208 bytes of state and completed one step in 0.176 s. A nonzero-gradient LoRA-B sentinel changed
by max absolute 1.0e-4 (L2 0.01810). Peak allocated/reserved VRAM remained 6,621,465,600/6,834,618,368 bytes;
host wall was 10.94 s, max RSS 5,774,996 KiB and swap zero. No checkpoint was created. This proves one-update
resource feasibility only. Before multi-step training, Phase 3B must freeze a deterministic 40/10 demonstration
train/validation pipeline and define checkpoint/validation cadence without using environment final states for tuning.

The first training smoke contract is now static PASS. The frozen 40/10 episode split remains 3,571/889 frames.
Because `evla-openvla` intentionally has no HDF5 dependency, `evla-core` exported a hash-frozen PNG/JSON subset:
20 deterministic distinct training frames and timestep 0 from each of the 10 validation episodes. The model process
will run validation at steps 0 and 20, perform 20 clipped AdamW updates, then save LoRA adapter, optimizer/RNG state,
resolved config and file hashes. The 10-frame validation is pipeline smoke only, not full offline validation.

Formal smoke v1 is FAIL despite exit 0: the generated module defined `main()` but omitted its invocation, so no model,
training, validation, output or checkpoint ran. Wall 1.33 s and max RSS 688,484 KiB corroborate silent non-execution.
V2 uses a new ID and an explicit wrapper `__main__` call; the model/data/optimizer contract is unchanged. Its static
entrypoint regression is PASS.

Formal smoke v2 is FAIL with CUDA OOM on the first training forward after the initial 10-frame validation. The OOM
requested 252 MiB with only 99.44 MiB free; PyTorch reported 6.08 GiB allocated and 201.87 MiB reserved/unallocated.
No optimizer step or checkpoint occurred, and initial validation metrics were not serialized. V3 must explicitly
delete per-validation logits/tensors, collect Python objects, empty the CUDA cache at the phase boundary, and use the
allocator's expandable-segments mode under a new ID. This is a memory-lifecycle fix, not a model/data change.

The user then accidentally reran v2 rather than v3. Because the project-provided guard used `test ! -e` without
`set -e`, it did not stop and overwrote the frozen v2 raw-log path. The rerun computation completed 20 finite clipped
updates and a loadable checkpoint: train first/last loss 22.7779/12.4246, first/last-10 means 14.9248/10.3290;
val10 loss 16.9151 to 8.8129 and token accuracy 4/70 to 8/70. Peak allocated/reserved VRAM was
6,866,165,248/7,291,797,504 bytes. Reused ID/log makes this attempt protocol FAIL and its checkpoint is quarantined.
V3 remains the required clean canonical run; future commands use errexit and explicit guard blocks.

Canonical v3 is PASS. GPU preflight showed only 733 MiB used. It completed all 20 clipped updates, val10 at steps
0/20, and a hash-verified adapter/optimizer/RNG checkpoint. Train loss first/last was 22.7779/12.3476 and first/last
10-step mean 14.9103/10.3167. Val10 loss fell 16.9151 to 8.8018 (47.96%) while token accuracy rose 4/70 to 9/70.
All gradients were finite; pre-clip norm ranged 360.34--486.92. Peak allocated/reserved VRAM was
6,755,235,328/7,128,219,648 bytes. Train plus two val10 passes took 18.55 s; host wall 26.99 s, max RSS 5,778,836
KiB, no swap. The checkpoint contains five verified files and CPU-loadable optimizer state at step 20. These results
prove the short training/checkpoint pipeline only; val10 is not full offline validation and neither loss nor token
accuracy is environment reward/success. Next is preregistered evaluation on all 889 frozen validation frames.

The full-validation gate is now static PASS. `evla-core` exported all 889 frames from the 10 frozen validation
episodes into a hash-verified PNG/JSON manifest; `evla-openvla` will load the canonical v3 step-20 adapter read-only.
It reports sample-weighted loss, action-token accuracy, continuous-action L1, per-episode metrics, timing and memory.
No training, checkpoint selection or environment execution is permitted by this gate.

Step-20 fullval889 is PASS over all 889 frames/6,223 action tokens. Mean/median loss is 8.9271/8.9149, token
accuracy 1,243/6,223 (19.97%), and mean/median continuous L1 0.2073/0.1927. Episode mean loss spans 8.0533--9.7417
and token accuracy 14.29%--23.96%. Validation took 206.17 s at 4.31 frames/s; inference median/P95 was
0.2261/0.2388 s. Peak allocated/reserved VRAM was 5,403,944,960/5,567,938,560 bytes; wall 213.96 s, max RSS
5,783,256 KiB, no swap. This is complete frozen offline coverage, but without a same-protocol base fullval run it
cannot establish improvement. It remains unrelated to environment reward/success.

A same-protocol base fullval889 evaluator is static PASS. It uses the identical 889-frame manifest, labels and metrics
without loading any adapter. This paired base run is required before interpreting the step-20 offline numbers or
granting any checkpoint eligibility. It remains read-only and performs no environment execution.

Base fullval v1 is FAIL before completing frame 1: it incorrectly used the PEFT wrapper path
`model.base_model.model` on a direct `OpenVLAForActionPrediction`. Model shards loaded, but no metric/output was
produced. V2 uses `wrapped=model` under a new ID and passes a no-CUDA structural regression. The terminal window
closed because shell errexit propagated the Python failure, not because the GPU or OS crashed.

Base fullval v2 is PASS: loss 16.9597, token accuracy 600/6,223 (9.64%), continuous L1 0.3077. Against the identical
889-frame base rows, step-20 reduces mean loss by 8.0326 (47.36%), increases token accuracy by 10.33 percentage
points/643 correct tokens (107.17% relative), and reduces L1 by 0.1004 (32.63%). Loss improves on 883/889 frames and
L1 on 732/889; all 10 episodes improve on loss, token accuracy and L1. This is sufficient to classify offline
validation improvement PASS and authorize a development-only environment gate. It does not prove reward/success,
does not select the checkpoint for final evaluation, and does not authorize held-out environment states.

The adapter-aware dependency-isolated single-episode development gate is static PASS. It reuses init state 0 and the
frozen Phase 3A coordinator, while a new `evla-openvla` worker hash-verifies and loads the canonical adapter before
`predict_action`. A formal PASS must separately report execution and policy success. One episode cannot authorize
dev5, held-out evaluation, checkpoint selection or further training.

The formal step-20 adapter single-development episode is PASS for both execution and policy success. Init0 succeeds
at step 102 with reward 1 and termination, versus the frozen base's step 85 on the same init. Inference median is
0.4540 s; peak allocated/reserved VRAM 5,403,944,960/5,567,938,560 bytes; episode end-to-end 65.17 s, host wall
71.85 s and zero swap. Both policies succeed; the +17 steps/+20% observation at n=1 cannot establish stable
efficiency regression. This authorizes only dev init0--4 evaluation.

Dev5 v1 is FAIL before worker readiness: adapter loading requested 2 MiB when only 23.94 MiB GPU memory was free.
No episode started, so this is external resource contention rather than policy failure. Wall 15.45 s, max RSS
5,768,284 KiB, zero swap. V2 preserves the compute contract under a new ID and adds a host preflight requiring at
least 8,000 MiB free before the coordinator can create an output directory.

Dev5 v2 is PASS for execution and 5/5 policy success. Adapter steps are [102,77,73,71,72] versus frozen base
[85,87,78,86,85]: totals 395 versus 421 and means 79.0 versus 84.2 (6.18% lower). Adapter is faster on four states
and slower by 17 steps on init0, so the aggregate improves but state-level superiority is false. Peak allocated/
reserved VRAM is 5,403,944,960/5,567,938,560 bytes; rollout 210.53 s, host wall 214.76 s, zero swap. Combined with
paired fullval889 improvement, this freezes step-20 for exactly one preregistered held-out init5--49 final. No tuning
or checkpoint change is allowed before that final.

The immutable heldout45 final is execution PASS and policy success 43/45 (95.56%; Wilson 95% 85.17%--98.77%).
Failures are init12 and init44, both at 600 steps/reward 0; base succeeded them in 593 and 68 steps respectively.
Adapter capped mean/median/P95 steps are 95.24/70/123.4 versus base 91.87/76/101.4. On the 43 common-success states,
adapter is faster on 32, slower on 10 and tied on one, with mean/median delta -9/-6 steps. Thus typical successful
control is faster, while two catastrophic tails reduce success from 45/45 to 43/45 and worsen capped mean/P95.

Phase 3B final conclusion: QLoRA engineering feasibility and paired offline adaptation PASS, but environment
non-regression FAIL. The frozen base remains the recommended reliable demo/deployment policy; the step-20 adapter is
a research artifact. Held-out outcomes are frozen and cannot be used for further tuning, rerun or checkpoint change.
