# Embodied VLA Lab continuation entry — Phase 3A single-image complete

Updated: 2026-08-27

This is the short continuation entry for a new conversation. The complete cumulative history remains:

`docs/handoffs/2026-08-26_phase1d_complete_total_project_handoff.md`

The next assistant must read that cumulative handoff from beginning to end, including its latest Sections 20--25,
then read this file. Earlier handoffs remain historical supporting material; the cumulative file plus this entry are
the authoritative starting point.

## Repository and operating boundary

- Project root: `/home/kxj/embodied_ws/embodied_vla_lab`.
- Git HEAD: `f0c0368fd07b51928bb1b96be2c7e0fc45818182`.
- Submodules: LeRobot `8b256a6c0d4769c3cc3e7e98f04940126398a391`; LIBERO
  `8f1084e3132a39270c3a13ebe37270a43ece2a01`; OpenVLA
  `c8f03f48af692657d3060c19588038c7220e9af9`.
- The worktree contains the current Phase 2/3 project assets as modified/untracked files. Preserve them. The unrelated
  pre-existing `env_backup_evla-core.yml` must not be added or modified automatically.
- Do not use multi-agent.
- Short local reads, hashes, CPU metadata/checkpoint loads, static checks, and project documentation/code writes are
  authorized. Long downloads, environment installation/mutation, GPU model loads, training, and longer environment
  evaluations must be supplied as one host command for the user to run and paste back.
- Do not automatically run GUI, W&B, Hub/upload, remote writes, commit, push, or deletion.
- Every new run needs a new experiment ID, raw log, output directory, structured result, and relevant manifest entry.
- For every pasted host output: first classify `PASS`, `FAIL`, or `EXPECTED_EXCEPTION`; then analyze all available
  loss/eval/success/reward/grad/throughput/VRAM/RSS/time/checkpoint/statistical evidence and its boundary; update
  structured results, manifests, README, roadmap, phase documentation, cumulative handoff and reproduction assets;
  finally provide exactly one smallest safe host command or stop explicitly at an authorization boundary.
- Offline validation loss must never be described as environment success or reward.

## Frozen completed evidence

- Phase 1B/1C local PushT finals are frozen: deterministic ACT 6/100, VAE ACT 2/100, Diffusion EMA 4/100.
- Phase 1D official PushT reference final is frozen at 63/100. It establishes that the evaluator can reproduce a
  strong public policy, but does not isolate which training-recipe difference caused the local baselines to be weak.
- Phase 2B ACT-on-LIBERO `turn_on_the_stove` is complete. The 100k training run PASSed; full offline validation has
  101 measurements over all 889 validation frames, minimum loss 0.426634 at step 1,000 and final 0.440030. Loss is
  not rollout success. Development selection compared steps 4,460, 49,060 and 100,000 and froze step 100,000.
- The frozen ACT step-100k checkpoint achieved 45/45 strict success and mean reward 1.0 on every held-out init state
  5--49. This is a complete finite-set result for the held-out states and only single-task/single-training-seed/
  fixed-runtime/unperturbed-vision evidence.

## Phase 3A latest state

- Target: official `openvla/openvla-7b-finetuned-libero-goal`, frozen revision
  `fa5ae1e7509348889295bba8e08621d8b55e9baf`.
- Separate `evla-openvla`: Python 3.10.21, PyTorch 2.11.0+cu128, Transformers 4.40.1, tokenizers 0.19.1,
  timm 0.9.10, bitsandbytes 0.47.0, accelerate 1.10.1. The NF4 `sm_120` kernel gate PASSed.
- The 16-file official snapshot is downloaded and hash-frozen: 15,085,050,797 logical bytes; four weight shards
  total 15,082,600,832 bytes. See `results/phase3a_openvla_checkpoint_manifest.yaml`.
- Import history is preserved. Avoid importing top-level `prismatic`, which drags in training-only RLDS/dlimp.
  Register only the four local HF inference classes. Do not install TensorFlow/RLDS/dlimp for inference.
- Single-image attempts v1--v3 are preserved as FAILs with distinct causes: remote `auto_map` resolution under strict
  offline; Accelerate calling forbidden `.to()` on a 4-bit model; and upstream `predict_action()` appending token
  29871 without extending the attention mask. None was OOM.
- v4 is PASS. `phase3a_openvla_4bit_single_image_v4_demo0_t000_seed20260827` loaded the frozen model in 5.806 s and
  generated one finite 7D action from the hash-frozen real LIBERO demo-0/t=0 agent-view frame in 1.024 s. Passive
  allocated/reserved VRAM: 4,720,663,552/4,836,032,512 bytes; peak: 5,060,689,920/5,196,742,656 bytes; process wall
  9.33 s; maximum RSS 5,826,412 KiB. Canonical result:
  `results/phase3a_openvla_4bit_single_image_v4.yaml`.
- The prompt alignment fix atomically appends token 29871 and one true mask entry. The scoped force-hooks shim avoids
  the Transformers 4.40.1 / modern Accelerate-bitsandbytes `.to()` mismatch. Preserve both until a separately
  controlled compatibility study replaces them.
- The 1.024 s value is one cold seven-token action observation (about 0.976 predictions/s), not steady-state latency,
  environment throughput, a 20 Hz controller, rollout success, or reward.
- No OpenVLA environment action has been executed. OpenVLA success, reward, termination behavior, closed-loop
  reliability and QLoRA/training feasibility remain unknown.

## Next smallest safe work

The dependency-isolated bridge is now implemented and its static contract is PASS:

1. keep LIBERO/MuJoCo environment ownership in `evla-core`;
2. keep OpenVLA model ownership in `evla-openvla`;
3. use a controlled local subprocess/IPC boundary without installing either full stack into the other environment;
4. freeze exactly one LIBERO task/init state and perform reset, one image transfer, one OpenVLA action prediction,
   frozen gripper adaptation, and exactly one environment `step`;
5. record observation/action shapes, finite checks, reward, success/done flags, inference and end-to-end latency,
   VRAM/RSS, hashes, and exception-safe process cleanup;
6. stop after one step. A PASS does not authorize a full episode. After that, separately gate one episode, then a
   small development rollout set, then a frozen held-out protocol. Do not begin QLoRA before rollout feasibility.

Static validation used CUDA-disabled request-only execution: no environment was constructed, no model was loaded and
no environment step occurred. The pending new-ID host gate is
`phase3a_openvla_libero_single_step_bridge_v1_init0_seed20260827`, fixed to init state 0 and seed 20260827. Its
canonical static record is `results/phase3a_openvla_single_step_bridge_static.yaml`.

The formal host gate subsequently PASSed. Exactly one online action was predicted and accepted on init state 0;
reward was 0, success/terminated/truncated were false, and cleanup confirmed `env_closed`. Model load/inference were
6.893/0.959 s, peak allocated/reserved VRAM was 5,060,688,896/5,196,742,656 bytes, process wall was 26.56 s and
maximum RSS was 5,829,496 KiB. This is interface evidence, not an episode success/failure or a success-rate result.
Canonical record: `results/phase3a_openvla_libero_single_step_bridge_v1.yaml`. The next work is a separately static-
validated persistent-worker single episode; batch rollout and QLoRA remain outside the current verified boundary.

The persistent-worker episode entrypoint is now implemented and static PASS. It is fixed to exactly one development
episode on init state 0, loads OpenVLA once, uses hash-bound per-step artifacts, stops by termination/truncation or 600
steps, and has exception-safe worker/environment cleanup. CUDA-disabled validation performed no model or environment
work. See `results/phase3a_openvla_single_episode_static.yaml`; formal host execution is pending.

The formal episode then PASSed with policy success on development init state 0 at step 85, sum/max reward 1.0.
Median/P95 inference were 0.4178/0.4367 s, peak allocated/reserved VRAM was
5,061,286,912/5,196,742,656 bytes, process wall was 59.08 s and maximum RSS was 5,827,332 KiB. Worker and environment
cleanup PASSed. N=1 proves feasibility, not a success rate. Canonical result:
`results/phase3a_openvla_libero_single_episode_v1.yaml`. Next is a frozen five-state development rollout only.

The five-state development entrypoint is now static PASS: init states 0–4, seeds 20260827–20260831, one persistent
model load and five episodes. No GPU/environment execution occurred in this static gate. See
`results/phase3a_openvla_dev5_static.yaml`; formal host execution remains pending.

The formal dev5 rollout then PASSed with 5/5 strict success on states 0–4, completion steps 85/87/78/86/85 and reward
1.0 for every episode. Median/P95 inference were 0.4125/0.4386 s; peak allocated/reserved VRAM was
5,061,286,912/5,196,742,656 bytes. This is development evidence with a wide Wilson interval, not final 100% evidence.
Canonical result: `results/phase3a_openvla_libero_dev5_v1.yaml`. Untouched states 5–49 are the candidate held-out final.

States 5–49 are now preregistered as the base-policy held-out final with seeds 20260901–20260945 and an immutable
protocol. Static validation PASSed without model/environment execution. See `results/phase3a_openvla_final45_static.yaml`.

The held-out final then PASSed with 45/45 strict success on states 5–49. Mean/median steps were 91.87/76, but state 12
required 593/600 and state 40 required 221. Median/P95 inference were 0.4090/0.4325 s; peak allocated/reserved VRAM
was 5,061,286,912/5,196,742,656 bytes. This is complete finite-set single-task evidence; Wilson 95% is approximately
92.13%–100% under Bernoulli generalization. Canonical result:
`results/phase3a_openvla_libero_final_heldout45_v1.yaml`. Phase 3A is complete; next is a no-install QLoRA contract.

Phase 3B source/data contract is now PASS. The first resource gate is rank-8 all-linear QLoRA, batch 1, one
forward/backward and no optimizer step, using a local non-RLDS fixture. Current `evla-openvla` lacks PEFT; only exact
`peft==0.11.1 --no-deps` is pending as a separate no-CUDA host environment gate.

PEFT 0.11.1 installation/import now PASSes with the frozen runtime unchanged and no CUDA/model work. A real LIBERO
single-sample rank-8 QLoRA resource gate is implemented and static PASS: one forward/backward, no optimizer step or
checkpoint. Formal GPU execution is pending. See `results/phase3b_openvla_qlora_resource_static.yaml`.

Resource v1 completed forward/backward assertions but FAILed at result writing because `output` path was shadowed by
the model output. No metrics were serialized, so it is not PASS. V2 changes only variable names, uses a new ID, and
passes a no-CUDA writer regression. See `results/phase3b_openvla_qlora_resource_v1.yaml` and
`results/phase3b_openvla_qlora_resource_v2_static.yaml`.

V2 formal one-sample forward/backward is now PASS: loss 17.8423, action-token accuracy 0/7, continuous-action L1
0.4807, finite gradient norm 539.24, forward/backward 0.622/0.463 s, peak allocated/reserved VRAM
6,621,465,600/6,851,395,584 bytes, wall 8.44 s and max RSS 5,835,192 KiB. These are only untrained single-sample
resource diagnostics. There was no optimizer step, checkpoint, offline validation, environment reward or success.
Canonical result: `results/phase3b_openvla_qlora_resource_v2.yaml`. The next gate is one clipped optimizer update
under a new ID, not formal training.

Optimizer gate v1 is FAIL at an insufficient post-step sentinel: it checked only the first trainable LoRA parameter,
which may validly remain unchanged on the first Gaussian-initialized LoRA step. The call reached `optimizer.step()`;
this is not OOM, but no metrics were serialized. V2 preserves the one-sample/one-step contract and checks a parameter
selected by a finite nonzero pre-clip gradient. V2 static regression PASS; formal execution pending. See
`results/phase3b_openvla_qlora_optimizer_v1.yaml` and `results/phase3b_openvla_qlora_optimizer_v2_static.yaml`.

Optimizer gate v2 formal execution is PASS: one finite clipped update, 218,256,208-byte AdamW state, 0.176 s step,
nonzero finite LoRA-B delta, peak allocated/reserved VRAM 6,621,465,600/6,834,618,368 bytes, wall 10.94 s and no
checkpoint. The single fixture loss/accuracy/L1 remains resource-only evidence. Next is a static deterministic 40/10
demonstration train/offline-validation/checkpoint contract, followed by a short smoke—not full training or rollout.

The dependency-isolated train20/val10 checkpoint smoke is static PASS. `evla-core` exported a frozen 20-train/10-val
PNG/JSON fixture from the unchanged 40/10 (3,571/889 frame) split; `evla-openvla` requires no HDF5 mutation. Formal
execution will validate at steps 0/20 and save adapter plus optimizer/RNG/config/hashes. Val10 is pipeline smoke only.

Train smoke v1 is FAIL despite exit 0 because the generated script omitted the `main()` invocation; no model,
validation, optimizer step, output or checkpoint occurred. V2 is a new immutable ID with an explicit wrapper call and
unchanged compute contract. Its no-CUDA entrypoint regression is PASS; formal execution pending.

Formal v2 is FAIL with CUDA OOM before training step 1, after initial val10. No metric was serialized and no
checkpoint exists. V3 adds explicit deletion/GC/CUDA-cache cleanup around validation and uses expandable allocator
segments; static PASS under a new ID with unchanged compute contract.

The next pasted command accidentally reran v2. Compute PASSed 20 steps and produced a loadable checkpoint, with
val10 loss 16.9151→8.8129, but the reused ID and overwritten frozen raw-log path make it protocol FAIL. The checkpoint
is quarantined. V3 remains pending; all future guards use shell errexit and explicit path checks.

Canonical v3 is PASS: 20 finite clipped updates, val10 loss 16.9151→8.8018, token accuracy 4/70→9/70, peak
allocated/reserved VRAM 6,755,235,328/7,128,219,648 bytes, and a five-file hash/load-verified checkpoint at step 20.
This is training-pipeline smoke only. Next is the complete frozen 889-frame offline validation; no environment
reward/success or checkpoint-selection claim exists.

The complete 889-frame frozen validation fixture/evaluator is now static PASS under a new ID. It uses the canonical
v3 step-20 adapter read-only and does not train or execute LIBERO. Formal full offline validation is pending.

Step-20 fullval889 is now PASS: loss 8.9271, token accuracy 19.97%, continuous L1 0.2073. A same-protocol base-only
fullval889 comparison is static PASS and pending; no improvement or environment claim is allowed before it.

Base comparison v1 FAILed before frame 1 due to retaining a PEFT wrapper access path on the direct base model. The
terminal closed because of shell errexit after Python exit 1. V2 fixes only that path under a new ID and is static PASS.

Base v2 fullval889 PASS enables paired comparison. Step-20 improves loss 47.36%, token accuracy by 10.33 points and
continuous L1 32.63%; all metrics improve across 10/10 episodes. This authorizes only a development environment gate,
not held-out evaluation or final checkpoint selection.

Step-20 adapter succeeds on development init0 at step 102 (base 85); n=1 cannot establish efficiency change. The
adapter init0--4 dev5 gate is static PASS and pending. Held-out/final selection remain outside the boundary.

Dev5 v1 FAILed before episode 1 because only 23.94 MiB GPU memory was free during adapter load. This is resource
contention, not policy evidence. V2 keeps the protocol and adds an 8,000 MiB free-memory preflight; static PASS.

Dev5 v2 is 5/5 success; total steps 395 versus base 421, with adapter faster on four states and slower on init0.
Step-20 is now immutably selected for one heldout init5--49 final; final static contract PASS and pending.

Phase 3B final is complete: execution PASS, adapter success 43/45 versus frozen base 45/45. Adapter is usually faster
on common successes but fails init12/init44 and has worse capped mean/P95. Engineering/offline improvement PASS;
environment non-regression FAIL. Base remains the reliable demo policy; no post-heldout tuning or rerun is allowed.

There is no currently recorded active Phase 3 training run in the project state summarized here. If the user starts
or finishes another run outside this record, treat its pasted terminal output as new evidence and do not assume its
classification or completion from this document.

## Message to paste into a new conversation

```text
请继续 Embodied VLA Lab 项目。不要使用 multi-agent。

项目根目录：
/home/kxj/embodied_ws/embodied_vla_lab

请先从头到尾完整读取这两份交接文档，第一份是覆盖项目所有历史阶段和历史对话的累计事实源，第二份是
最新续聊入口：

/home/kxj/embodied_ws/embodied_vla_lab/docs/handoffs/2026-08-26_phase1d_complete_total_project_handoff.md
/home/kxj/embodied_ws/embodied_vla_lab/docs/handoffs/2026-08-27_phase3a_openvla_single_image_complete_resume.md

读取后，先执行文档要求的只读核对，并向我简要报告：Git HEAD、三个submodule pins、工作区保护边界、
Phase 1/2冻结结论、Phase 3A v1-v4失败/修复链、v4单图PASS的显存/时间/动作和统计边界，以及下一步
dependency-isolated单环境step bridge方案。不要因为读取交接而自动启动任何训练、评测、下载或安装。

短时本地读取、搜索、哈希、CPU加载、静态校验和项目文件写入可直接执行。正式训练、较长GPU推理、
较长环境评测、下载大文件和环境安装/变更，由你准备一条完整宿主命令，我手动运行并粘贴完整输出。
不要运行GUI、W&B、Hub/upload、commit、push或删除，除非我另行明确授权。不要修改或覆盖任何冻结
Phase 1/2 final、Phase 3A历史失败结果、raw log或已有实验目录；每次新运行必须使用新ID、日志、输出
目录、结构化结果和manifest条目。

我接下来会继续粘贴宿主终端输出。每次收到后请：
1. 首先判定PASS、FAIL或EXPECTED_EXCEPTION；
2. 分析输出中所有可用的loss、offline validation、success、reward、grad、吞吐、显存、主存、时间、
   checkpoint完整性和统计结论边界；不要把offline loss当作环境success或reward；
3. 更新结构化结果、checkpoint/model manifest、README、roadmap、phase文档、累计handoff和可复现资产；
4. 最后只给下一条最小、安全、独立的宿主命令，或明确停在授权边界。

当前最新已验证边界是OpenVLA NF4 4-bit真实LIBERO单图动作预测PASS；尚未执行OpenVLA环境step，下一步
应先开发并静态验证evla-core环境进程与evla-openvla模型进程之间的隔离单step bridge，不能直接开始
完整episode、批量rollout或QLoRA训练。
```
