# Embodied VLA Lab — Phase 1B Formal Training In Progress Handoff

更新时间：2026-08-25

项目根目录：`/home/kxj/embodied_ws/embodied_vla_lab`

这是当前最新 handoff。正式100k ACT PushT 训练正在宿主终端运行；新对话首先读取本文件，然后等待
用户粘贴最终输出。不要重复启动训练，不要删除或修改正在写入的 raw log/output/checkpoint。

## 1. Git 与授权边界

当前正式 `HEAD`：

```text
16232e3833d4a225b903ed60f2d7062ebaa68690
16232e3 docs: record phase 1b act smoke result
```

三个 submodule pins：

```text
lerobot  8b256a6c0d4769c3cc3e7e98f04940126398a391
libero   8f1084e3132a39270c3a13ebe37270a43ece2a01
openvla  c8f03f48af692657d3060c19588038c7220e9af9
```

本轮所有新增/修改均未提交。不要把这些本地项目资产视为异常，也不得未经用户单独授权 commit/push：

```text
 M experiments/act_baseline/README.md
?? configs/train/act_pusht_phase1b_formal_100k.yaml
?? configs/train/act_pusht_phase1b_fullsplit_batch8_100step.yaml
?? configs/train/act_pusht_phase1b_overfit_ep51.yaml
?? docs/handoffs/2026-08-25_phase1b_smoke_complete_overfit_handoff.md
?? docs/handoffs/2026-08-25_phase1b_formal_training_in_progress_handoff.md
?? results/phase1b_act_pusht_fullsplit_batch8_100step.yaml
?? results/phase1b_act_pusht_overfit_ep51_2000step.yaml
?? scripts/check_phase1b_act_checkpoint_accuracy.py
?? scripts/check_phase1b_act_formal_config.py
?? scripts/check_phase1b_act_fullsplit_batch8_config.py
?? scripts/check_phase1b_act_overfit_config.py
```

继续严格遵守：安装、dataset/model下载、任何新训练、GUI、W&B、Hub/upload、commit、push 都需要分别
明确授权。本次100k训练授权已消费，只覆盖当前正在运行的命令。不要使用多 agent。

同类只读源码/日志/status/hash 查看已授权助手自行执行，使用 `conda run -n evla-core` 或宿主已激活的
`evla-core` 环境。

## 2. 已完成阶段

- Phase 0：环境验证完成并提交。
- Phase 1A：固定 PushT v3 snapshot、数据/processor/action-window/padding/split contract 完成并提交。
- Phase 1B：config preflight、2-step BF16 smoke、单 episode overfit诊断、checkpoint reload/inference、
  fixed-data accuracy、完整 split batch-8 100-step resource gate 均已完成。
- Phase 1B正式100k训练正在运行，尚不能下最终结论。

固定 dataset：

```text
lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19
206 episodes / 25,650 frames / 10 Hz
train: 185 episodes / 23,043 frames
validation: 21 episodes / 2,607 frames
video backend: pyav
```

固定 validation IDs：

```text
[4, 8, 16, 20, 22, 39, 40, 41, 42, 43, 61, 65, 67, 96, 115, 119, 138, 150, 156, 182, 199]
```

## 3. 本轮 overfit 诊断

配置：`configs/train/act_pusht_phase1b_overfit_ep51.yaml`

- train episode 51 / validation episode 41，均73 frames；
- 2000 steps、batch 1、BF16、chunk 50、execute 10；
- 训练执行与最终 checkpoint：PASS；
- 预声明严格 overfit threshold：NO-GO。

关键训练数值：

```text
first 5 window L1 mean: 0.5182
last 5 window L1 mean: 0.1584
reduction: 69.43%
required: <=0.10 and >=80% reduction (not met)
wall: 97.33 s
peak PyTorch allocated: 0.94 GiB
exit: 0
```

Raw log：

```text
results/raw/phase1b_act_pusht_overfit_ep51_2000step.log
426 lines / 88,267 bytes
sha256 202d61be490fe96ccac2983bf44e875e37e7a994da873fac14943530fc904645
```

Checkpoint：

```text
outputs/train/phase1b_act_pusht_overfit_ep51_bf16/checkpoints/002000
model sha256 648d8a6f36db4c8762048686d93f9899261fa69c7096ab486c85608d6471a2f9
```

Checkpoint reload/inference PASS：`[1,50,2]` normalized/unnormalized chunks finite，peak 222.39 MiB。
日志 SHA-256：`a0972d4907ecf06f64eb0c869927ea1d406ddd5a5e7c66ceaaff28b5dae26958`。

Fixed-data deployment-path accuracy PASS：

| Metric | Train ep.51 | Validation ep.41 |
|---|---:|---:|
| full chunk MAE | 17.1033 px | 76.7296 px |
| first 10 MAE | 20.7338 px | 58.5615 px |
| first action MAE | 19.9984 px | 51.9059 px |

Train/validation full-chunk ratio `0.222904`，证明明显单 episode记忆，但绝对误差仍明显；不继续围绕单
episode调参。Accuracy raw log SHA-256：
`ee2ce7a45a57a08b7535fb899b35a95467f8bd9bc22129ec41c788bcd8644272`。

结构化总结果：`results/phase1b_act_pusht_overfit_ep51_2000step.yaml`。

## 4. 完整 split batch-8 资源 gate

配置：`configs/train/act_pusht_phase1b_fullsplit_batch8_100step.yaml`

严格离线 preflight PASS。经授权训练100 steps，结果 PASS：

```text
batch/effective batch: 8/8
first window total/L1/KL: 35.916 / 0.832 / 3.508
final window total/L1/KL: 4.486 / 0.597 / 0.389
pre-clip grad norm: 668.624 -> 125.571
step-100 capped eval loss (256 frames): 0.4841
warm-window mean: 0.06022 s/step / 132.78 samples/s
peak PyTorch allocation: 0.94 GiB
whole wall: 13.43 s
exit: 0
```

Raw log：

```text
results/raw/phase1b_act_pusht_fullsplit_batch8_100step.log
413 lines / 21,410 bytes
sha256 874cbd7c5fcbf23f07c7e0c1a72ff58df986f353f3afcf93fe1426864e9fcf26
```

结构化结果：`results/phase1b_act_pusht_fullsplit_batch8_100step.yaml`。

## 5. 正式100k配置与 preflight

正式配置：`configs/train/act_pusht_phase1b_formal_100k.yaml`

```text
steps: 100,000
batch/effective batch: 8/8
effective epochs: 34.7177
precision: BF16
chunk/action steps: 50/10
optimizer: AdamW lr 1e-5, weight decay 1e-4, grad clip 10
scheduler: None
pretrained backbone: None
log_freq: 200
eval_steps: 5,000; full 2,607 validation frames
save_freq: 25,000
checkpoints: 25k / 50k / 75k / 100k
W&B/Hub/env eval/recording/EMA: disabled
```

Checker：`scripts/check_phase1b_act_formal_config.py`。它与已通过100-step gate逐字段比较，只允许
output/job、steps、log/eval/save cadence和checkpoint enable变化。

首次 formal preflight 因 checker 在 `draccus.parse()` 前漏导入 `ACTConfig` 而 FAIL；发生在 dataset
factory前，无副作用。补导入后同一严格离线命令 retry PASS。TorchCodec→PyAV和审计沙箱无GPU导致
runtime config临时回退CPU均为已知预期提示；YAML CUDA/BF16意图固定。

## 6. 当前正在运行的正式训练

用户已单独授权并在宿主启动：

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab &&
conda activate evla-core &&
test ! -e outputs/train/phase1b_act_pusht_formal_100k_bf16 &&
test ! -e results/raw/phase1b_act_pusht_formal_100k.log &&
set -o pipefail

/usr/bin/time -v \
env -u ALL_PROXY -u all_proxy \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 \
  WANDB_MODE=disabled \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONUNBUFFERED=1 \
  lerobot-train \
  --config_path=configs/train/act_pusht_phase1b_formal_100k.yaml \
  2>&1 | tee results/raw/phase1b_act_pusht_formal_100k.log

train_status=${PIPESTATUS[0]}
sha256sum results/raw/phase1b_act_pusht_formal_100k.log
wc -l -c results/raw/phase1b_act_pusht_formal_100k.log
printf 'TRAIN_EXIT_STATUS=%s\n' "$train_status"
```

创建本 handoff 时的只读快照：训练仍在运行，约 step 42,123 / 100,000（42%），elapsed约45分钟；
近期窗口约：

```text
loss 0.179-0.185
l1_loss 0.139-0.149
kld_loss 0.004
pre-clip grad norm 8.9-9.5
step time ~0.062 s
throughput ~129 samples/s
peak allocation 0.94 GiB
```

25k checkpoint 已存在：

```text
outputs/train/phase1b_act_pusht_formal_100k_bf16/checkpoints/025000
outputs/train/phase1b_act_pusht_formal_100k_bf16/checkpoints/last -> 025000
```

以上只是运行中快照，不是最终结论。不要在新对话中重复启动、resume、kill、删除或修改训练。先让用户
粘贴最终终端输出；必要时可自行只读读取 raw log/output/checkpoints。

## 7. 收到最终输出后的强制处理顺序

1. 首先判定：训练执行 PASS/FAIL/预期例外。
2. 核对 `TRAIN_EXIT_STATUS` 与 `/usr/bin/time` exit status。
3. 核对100k完成、20次 eval、25k/50k/75k/100k checkpoint和 `last -> 100000`。
4. 解析全部500个200-step train窗口以及20个 eval loss：首/末、min、趋势、NaN/Inf、grad、吞吐、显存。
5. 核对 raw log lines/bytes/SHA-256、checkpoint tree和关键文件SHA-256。
6. 更新 `experiments/act_baseline/README.md`，新增小型结构化 result YAML；不要提交大 raw log。
7. 给出下一条最小、安全宿主命令。下一 gate应先做 checkpoint reload与固定数据 accuracy，再设计
   PushT rollout protocol；不要直接宣称 success/generalization。

若训练失败，不要自动删除或重跑；先保留已有 checkpoint/log并诊断。任何 resume/retrain仍需新授权。

## 8. 已知预期例外与边界

- TorchCodec FFmpeg/CXXABI loader warning后fallback显式PyAV；不安装/修改TorchCodec或FFmpeg。
- Image stats count 19,619是每episode最多100帧抽样，不是缺损。
- OpenCV metadata warning保持headless runtime，不安装非headless包。
- PushT上游384×384声明与gym runtime 96×96差异已固定为项目96×96 config。
- `next.success`全false，不能作为环境success metric。
- Training log的grad norm是clipping前值，threshold 10。
- Offline eval只有总loss，不含L1/KL分项，也不是环境rollout。
- 所有训练/推理性能数字只适用于当前RTX 5070 Ti Laptop GPU和当前配置。

## 9. 新对话启动模板

把下面内容作为新对话第一条消息发送：

```text
请继续 Embodied VLA Lab 项目。

项目根目录：
/home/kxj/embodied_ws/embodied_vla_lab

请先完整读取最新 handoff：
/home/kxj/embodied_ws/embodied_vla_lab/docs/handoffs/2026-08-25_phase1b_formal_training_in_progress_handoff.md

然后只读核对 git status、HEAD、三个 submodule pins，并读取：
- configs/train/act_pusht_phase1b_formal_100k.yaml
- results/phase1b_act_pusht_fullsplit_batch8_100step.yaml
- results/phase1b_act_pusht_overfit_ep51_2000step.yaml
- experiments/act_baseline/README.md

正式100k训练已经在上一对话获得授权并启动。不要重复启动、resume、kill或删除训练；先接收我粘贴的
最终宿主输出，并可自行只读检查：
- results/raw/phase1b_act_pusht_formal_100k.log
- outputs/train/phase1b_act_pusht_formal_100k_bf16/

收到输出后必须：
1. 先判定 PASS、FAIL或预期例外；
2. 解释 loss/eval/grad/吞吐/显存/时间和结论边界；
3. 核对并更新结构化结果、checkpoint manifest、README与可复现资产；
4. 再给下一条最小、安全宿主命令。

安装、dataset/model下载、任何新训练、GUI、W&B/Hub/upload、commit、push仍需分别明确授权。
同类只读查看可自行执行并使用evla-core环境。不要使用多 agent。
```
