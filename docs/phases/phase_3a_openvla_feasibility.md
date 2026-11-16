# Phase 3A — OpenVLA 4-bit inference feasibility

Updated: 2026-08-27  
Current gate: static, isolated environment/NF4 kernel, and frozen checkpoint download/hash PASS; model not loaded.

## Objective

Establish whether the official LIBERO-Goal-finetuned OpenVLA 7B checkpoint can run safely and reproducibly on the
12 GB RTX 5070 Ti Laptop GPU before attempting any QLoRA training. The target is the suite-aligned checkpoint
`openvla/openvla-7b-finetuned-libero-goal`, not the zero-shot base checkpoint.

## Frozen model contract

- official repository revision: `fa5ae1e7509348889295bba8e08621d8b55e9baf`;
- pinned source submodule: `c8f03f48af692657d3060c19588038c7220e9af9`;
- approximately 8B BF16 parameters / 15.1 GB logical repository;
- upstream fine-tuning: LoRA rank 32, 60k steps, batch 128 on 8 A100-80GB GPUs;
- LIBERO-Goal action statistics selected with `unnorm_key=libero_goal`;
- seven generated action tokens are decoded through 256 bins and q01/q99 statistics;
- 224x224 agent-view image with the inference-time center 90% crop;
- no proprioception and no wrist image in the official OpenVLA policy path;
- gripper output is mapped from `[0,1]` to `[-1,1]`, binarized, and sign-inverted for LIBERO.

This is not a pure architecture comparison with Phase 2B ACT. ACT used two 128x128 cameras, 9D proprioception,
action chunks and project-trained single-task data; OpenVLA uses one 224x224 view, language, no state, token-by-token
actions and an official suite-level fine-tuned checkpoint.

## Runtime decision

`evla-core` must remain unchanged. OpenVLA's Transformers 4.40.1/tokenizers 0.19.1 stack conflicts with LeRobot's
current stack. Create a separate `evla-openvla` Python 3.10 environment with Blackwell-compatible cu128 PyTorch.
Use bitsandbytes 0.47.0 NF4 4-bit; its CUDA 12.8 Linux wheel includes `sm_120`. Do not install the upstream pinned
PyTorch 2.2, TensorFlow/RLDS training stack, FlashAttention 2.5.5, W&B, or OpenVLA dependencies wholesale.

## Ordered gates

1. Create the isolated minimal environment and verify package imports plus CUDA/BF16/NF4 kernel support.
2. Query/freeze the official snapshot and download its approximately 15.1 GB payload into a dedicated cache.
3. Verify every downloaded file size/hash and inspect config, processor, tokenizer and `libero_goal` statistics.
4. Load 4-bit with eager/SDPA attention and measure passive/active VRAM, RSS and latency on one frozen local image.
5. Run one action/environment-step smoke, then one episode, before any multi-episode evaluation.
6. Only after inference and rollout pass, attempt a batch-1 QLoRA forward/backward resource gate.

Environment creation, package installation, model download, GPU model load, inference and rollout are deliberately
separate host gates. QLoRA formal training is not authorized by an inference PASS.

## Static gate result

The source-only checker PASSed against the pinned OpenVLA submodule. It confirmed the official LIBERO-Goal checkpoint
identity, 7D q01/q99 action path, center crop, prompt, 4-bit loader and gripper adapter. No target environment or model
cache currently exists. The next host gate creates `evla-openvla` and runs only a tiny BF16/NF4 CUDA kernel smoke;
it does not download or load OpenVLA.

The isolated environment gate PASSed on the host. PyTorch 2.11.0+cu128 and bitsandbytes 0.47.0 executed a finite
NF4 `Linear4bit` kernel on the RTX 5070 Ti (`sm_120`). Peak allocated/reserved VRAM was only
8,527,360/23,068,672 bytes for this tiny kernel. Environment creation took 1,358.22 s, maximum RSS was 1,341,168 KiB,
and the environment occupies 7.2 GiB.

The frozen official snapshot download gate subsequently PASSed. All 16 files (15,085,050,797 logical bytes,
including four weight shards totaling 15,082,600,832 bytes) were hashed at exact revision
`fa5ae1e7509348889295bba8e08621d8b55e9baf`. Download took 4,027.64 s, process wall was 4,027.80 s and maximum RSS
was 487,816 KiB; 56,123,138,048 filesystem bytes remained. Config inspection confirmed 224x224 input, seven action
dimensions, 256 bins and `libero_goal` q01/q99 statistics over 52,042 transitions / 428 trajectories. No model was
constructed, CUDA was not initialized, and there is no VRAM, action, latency, rollout, success/reward or training
evidence yet. The next gate is one NF4 4-bit model load and one frozen-image action prediction.

The first local-source import host attempt installed the two frozen lightweight dependencies successfully but then
FAILed because Python executed `prismatic/__init__.py`, which eagerly imported training-only RLDS code and required
`dlimp`. Installing the TensorFlow/RLDS stack is intentionally out of scope for inference. The checker and inference
entrypoint now register only the `prismatic.extern.hf` namespace paths. A final v3 import-only rerun PASSed all four
required classes in 2.21 s with maximum RSS 811,788 KiB, correct draccus/rich versions 0.8.0/13.9.4, no model load,
and `cuda_initialized=false`. The intermediate v2 import was functionally successful but is invalid as the canonical
record because it read a stale package `__version__`; v3 uses installation metadata.

The first 4-bit single-image host attempt then FAILed before processor construction. Although local classes were
registered, `trust_remote_code=true` followed the checkpoint's `auto_map` reference to the base Hub repository and
could not obtain `processing_prismatic.py` in strict-offline mode. It exited after 2.44 s with maximum RSS 834,704
KiB; no weights were loaded and CUDA was not initialized. The v2 entrypoint keeps the explicit pinned local class
registrations but sets `trust_remote_code=false`, preventing any remote-code resolution.
The corrected strict-offline processor-only check PASSed with `PrismaticProcessor`, `PrismaticImageProcessor` and
`LlamaTokenizerFast`; CUDA remained uninitialized and no weights were read. The full v2 host model-load gate is next.

The v2 host attempt read all four quantized checkpoint shards in about 10 s, then FAILed during final device dispatch.
Accelerate 1.10.1 treated modern bitsandbytes 0.47 as movable and called `model.to(device)`, but frozen Transformers
4.40.1 rejects `.to()` for any 4-bit model. This was not OOM: process wall was 13.91 s and maximum RSS 5,833,868 KiB.
No usable model was returned and no inference ran. The v3 script applies a narrowly scoped dispatch shim that forces
Accelerate hooks and avoids `.to()` without changing any frozen package version or weight.

The v3 dispatch shim worked: all shards loaded, a usable model was returned, and generation entered the first Llama
forward. It then FAILed with attention logits length 276 versus causal-mask length 275. Processor-only measurement
proved the exact cause: the prompt produced 19 IDs and a 19-entry mask, ending in token 29901; upstream
`predict_action()` appended required token 29871 only to `input_ids`. With 256 visual patches, inputs became
20+256=276 while the mask became 19+256=275. This is independent of VRAM and NF4 numerical kernels. The v4 caller
appends token 29871 and one true mask entry together, causing the upstream conditional append to be skipped.

The corrected v4 single-image gate PASSed. The checkpoint loaded in 5.806 s and generated one finite 7D action from
the frozen real `demo_0/t=0` agent-view frame in 1.024 s. Passive allocated/reserved VRAM was
4,720,663,552/4,836,032,512 bytes and peak was 5,060,689,920/5,196,742,656 bytes (about 4.71/4.84 GiB peak).
Process wall was 9.33 s, maximum RSS 5,826,412 KiB, and RSS at result emission was 1,882,896 KiB. The raw gripper
component 0.0 became LIBERO command +1.0 after the frozen normalize/binarize/invert adapter. The bitsandbytes
FutureWarning concerns a future PyTorch internal API removal and did not affect this finite PASS.

This is one cold action generation only: observed 0.976 predictions/s is neither a steady-state latency distribution
nor LIBERO environment throughput, and is not a demonstrated 20 Hz controller. No action was executed, so success,
reward and closed-loop behavior remain unknown.

## Dependency-isolated single-step bridge static gate

The bridge static gate is PASS. `evla-core` retains exclusive ownership of LIBERO/MuJoCo and invokes a dedicated
`evla-openvla` worker through a local subprocess. A versioned JSON request binds the experiment, checkpoint revision,
prompt, rotated 128x128 uint8 agent-view PNG and its SHA-256. The worker returns a hash-bound response containing raw
and LIBERO-adapted 7D actions plus load/inference/VRAM/RSS measurements. The coordinator checks identity, shape,
finite values, `[-1,1]` bounds and binary gripper command before permitting exactly one `env.step()`.

The image is explicitly rotated 180 degrees, matching upstream OpenVLA LIBERO evaluation, before the already frozen
90% center crop. The gripper path remains normalize, binarize, then invert. Output directories and IPC artifacts are
no-overwrite; the environment is closed in `finally`, and the worker has a 180-second timeout. Static validation
executed the request-only worker path with CUDA disabled: no environment construction, model load or environment step
occurred. Canonical static result: `results/phase3a_openvla_single_step_bridge_static.yaml`.

The pending formal gate is `phase3a_openvla_libero_single_step_bridge_v1_init0_seed20260827`: fixed task
`turn_on_the_stove`, init state 0 and evaluator seed 20260827, exactly one reset/prediction/step. A PASS will establish
interface feasibility only. It will not authorize an episode or establish success rate, reward performance,
steady-state throughput, closed-loop reliability, loss, gradients or training feasibility.

## Dependency-isolated single-step formal result

`phase3a_openvla_libero_single_step_bridge_v1_init0_seed20260827` is PASS. On fixed init state 0, `evla-core`
reset the environment once and supplied one rotated/hash-bound online agent-view image to the isolated
`evla-openvla` worker. The frozen checkpoint loaded all four shards in 6.893 s and returned one finite in-range 7D
action in 0.959 s. Raw gripper 0.99608 became -1.0 through the frozen adapter. LIBERO accepted exactly one action in
0.0104 s; reward was 0, success/terminated/truncated were all false, and `env.close()` was confirmed.

Passive allocated/reserved VRAM was 4,720,663,552/4,836,032,512 bytes and peak was
5,060,688,896/5,196,742,656 bytes. Internal end-to-end time was 19.452 s; full process wall was 26.56 s and maximum
RSS was 5,829,496 KiB. There were no swaps or socket messages. The robosuite macro warning and bitsandbytes future
warning were non-semantic; the LIBERO helper confirmed the existing local asset cache and `/usr/bin/time` recorded
zero network traffic.

Reward 0 and success false after the first action are expected observations, not an episode failure and not a
success-rate estimate. No loss, offline validation, gradients, optimizer throughput or checkpoint was produced.
Canonical result: `results/phase3a_openvla_libero_single_step_bridge_v1.yaml`. Next, implement a persistent model
worker and statically validate exactly one full episode entrypoint before any additional host GPU/environment run.

That persistent-worker single-episode static gate is now PASS. The new worker loads the frozen model once, reports a
ready event, and serves hash-bound per-step requests over a JSON-lines control channel while request, image and
response artifacts remain on disk. The coordinator is frozen to one development episode on init state 0, stops on
termination/truncation or the 600-step horizon, writes a flushed trajectory JSONL, and records inference distribution,
reward, success, VRAM/RSS and artifact hashes. Cleanup first requests graceful shutdown, then terminates/kills a
stuck worker, and always closes LIBERO. CUDA-disabled request-only validation PASSed without model/environment work.
Canonical static record: `results/phase3a_openvla_single_episode_static.yaml`. The formal host episode remains pending.

The formal single episode is PASS for both execution and policy outcome. On development init state 0, OpenVLA
reached strict success at step 85 with sum/max reward 1.0 and terminated without truncation. All 85 actions were
finite and in range. The gripper adapter emitted -1.0 for 64 steps and +1.0 for 21. The model loaded once in 5.781 s;
warm inference median/P95 were 0.4178/0.4367 s, mean 0.4214 s, and pure inference throughput was 2.37 predictions/s.
Peak allocated/reserved VRAM was 5,061,286,912/5,196,742,656 bytes; process wall was 59.08 s and maximum RSS was
5,827,332 KiB. Cleanup confirmed both environment close and worker reap.

This one successful development episode proves closed-loop task completion feasibility, but N=1 is not a success-rate
estimate and gives no generalization or comparison claim. No loss, offline validation, gradients, optimizer throughput,
checkpoint creation or training occurred. Canonical result: `results/phase3a_openvla_libero_single_episode_v1.yaml`.
The next bounded gate is the frozen development set init states 0–4 using one persistent model load.

That dev5 gate is now implemented and static PASS. It is frozen to init states 0–4 and evaluator seeds
20260827–20260831, with five episodes, one persistent model load and a 600-step cap per episode. Each episode records
success independently; completed evaluation execution remains distinct from policy success. No held-out state,
checkpoint selection or training is included. Canonical static record: `results/phase3a_openvla_dev5_static.yaml`.

The formal dev5 rollout is PASS and achieved 5/5 strict successes on init states 0–4. Completion steps were
85/87/78/86/85 (mean 84.2), and every sum/max reward was 1.0. Across 421 predictions, inference median/P95 were
0.4125/0.4386 s and pure throughput was 2.42 predictions/s. Peak allocated/reserved VRAM was
5,061,286,912/5,196,742,656 bytes; process wall was 211.38 s and maximum RSS was 5,835,520 KiB. Cleanup PASSed with
no swaps or socket traffic.

This is bounded development evidence. If treated as Bernoulli samples, 5/5 has a Wilson 95% interval of approximately
56.6%–100%; it is not a final 100% claim. No training, loss, offline validation, gradients or checkpoint creation
occurred. Canonical result: `results/phase3a_openvla_libero_dev5_v1.yaml`. The untouched states 5–49 are now the
candidate held-out finite set and must be preregistered before execution.

The base-policy final protocol is now preregistered and static PASS. It completely enumerates init states 5–49,
which were untouched by OpenVLA development, with seeds 20260901–20260945, one persistent model load and a 600-step
horizon. States 0–4 are explicitly excluded. The protocol cannot be revised using final outcomes. Static validation
performed no model/environment work. Canonical record: `results/phase3a_openvla_final45_static.yaml`.

The preregistered held-out final is PASS: all 45 untouched init states 5–49 reached strict success with sum/max reward
1.0. This used 4,134 predictions. Mean/median completion steps were 91.87/76, range 65–593 and P95 101.4. Init state
12 required 593/600 steps and state 40 required 221, exposing a meaningful efficiency/horizon-margin tail despite
universal success. Inference median/P95 were 0.4090/0.4325 s and pure throughput was 2.44 predictions/s. Peak
allocated/reserved VRAM was 5,061,286,912/5,196,742,656 bytes; process wall was 1,844.55 s and maximum RSS was
5,768,988 KiB. Cleanup, no-swap and no-network checks PASSed.

This completely enumerates the frozen 45-state finite held-out set. If generalized as Bernoulli samples, the Wilson
95% interval is approximately 92.13%–100%. It remains single-task, fixed-checkpoint/runtime and unperturbed-vision
evidence, not broader generalization. No loss, gradients, optimizer, training or checkpoint creation occurred.
Canonical result: `results/phase3a_openvla_libero_final_heldout45_v1.yaml`. Phase 3A base-policy feasibility is complete;
the next phase is a separately gated QLoRA resource study, not immediate formal training.
