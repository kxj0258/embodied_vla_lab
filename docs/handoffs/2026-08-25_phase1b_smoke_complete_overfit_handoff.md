# Embodied VLA Lab — Phase 1B Smoke Complete / Overfit Planning Handoff

更新时间：2026-08-25

项目根目录：`/home/kxj/embodied_ws/embodied_vla_lab`

本文件是当前最新 handoff。创建前 `HEAD` 为 `16232e3` 且工作树 clean；本 handoff 是用户要求生成的
本地总结文件，当前未提交，不应把它视为异常，也不得未经授权 commit 或 push。

## 1. 当前阶段结论

- Phase 0 环境验证已完成并提交。
- Phase 1A PushT Dataset v3 数据、split、ACT processor/action-window contract 已完成并提交。
- Phase 1B 的 trainer/config 源码 preflight、严格离线 config/dataset preflight 和2-step BF16
  training smoke 均已 PASS。
- 2-step smoke 证明固定数据到 ACT forward/backward、gradient clipping 和 AdamW optimizer step 的
  最小训练链路可执行。
- 尚无 overfit/convergence 结论、checkpoint、rollout、success rate 或正式吞吐 benchmark。
- 下一阶段先只读设计 overfit config；任何更多训练仍需用户单独明确授权。

## 2. 正式提交与 Git 边界

当前正式提交链：

```text
16232e3 docs: record phase 1b act smoke result
3701124 feat: add phase 1b act smoke preflight
94e2db8 feat: complete phase 1a pusht data validation
4d67e21 docs: complete phase 0 environment validation
```

本轮没有 push。创建本 handoff 前工作树为：

```text
## main
```

本 handoff 创建后应只新增：

```text
?? docs/handoffs/2026-08-25_phase1b_smoke_complete_overfit_handoff.md
```

不要未经单独授权 stage/commit 本文件。

三个 submodule pins 未改变：

```text
lerobot  8b256a6c0d4769c3cc3e7e98f04940126398a391
libero   8f1084e3132a39270c3a13ebe37270a43ece2a01
openvla  c8f03f48af692657d3060c19588038c7220e9af9
```

## 3. 用户授权与操作边界

继续严格遵守：

- 安装任何依赖；
- 下载 dataset；
- 下载模型或 pretrained weights；
- 运行训练；
- 启动 GUI；
- W&B login/run；
- 上传 Hub/W&B；
- commit；
- push；

以上每类动作都需要用户分别明确授权，不能把一次授权扩展到其他动作。

本轮已经使用并消耗的授权：

- 下载固定 PushT snapshot；
- 提交 Phase 1A；
- 提交 Phase 1B preflight；
- 运行严格限定的2-step Phase 1B training smoke；
- 提交2-step smoke 结构化结果。

这些授权不自动延续到 overfit、checkpoint、rollout、W&B、上传或下一次 commit。

用户要求继续采用宿主执行模式：助手给出最小、安全命令，用户在终端运行并粘贴完整输出。每次收到
输出后必须先判定 `PASS`、`FAIL` 或预期例外，再解释结果与边界，更新项目记录，最后给下一条最小
安全宿主命令。不要使用多 agent。

## 4. Phase 0 固定环境

Phase 0 正式提交：

```text
4d67e21 docs: complete phase 0 environment validation
```

核心环境：

```text
conda env: evla-core
Python: 3.12.14
PyTorch: 2.11.0+cu128
torchvision: 0.26.0+cu128
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
compute capability: sm_120
BF16: available
OpenCV runtime: opencv-python-headless 4.13.0.92 only
```

三条已接受的 OpenCV metadata 例外：

- `gym-pusht` 声明需要 `opencv-python`；
- `robosuite` 声明需要 `opencv-python`；
- `hf-libero` 声明需要 `opencv-python`。

运行时只保留 headless OpenCV，不能为消除 metadata warning 安装非 headless OpenCV。

PushT 例外：pinned LeRobot 上游声明 384×384，但 `gym-pusht==0.1.6` runtime 是 96×96；项目固定
使用 `configs/env/pusht_96.yaml`，不修改上游 submodule。

Hugging Face 代理 workaround：需要联网 metadata/download 时，只在单条命令中取消
`ALL_PROXY`/`all_proxy`，保留 HTTP/HTTPS proxy：

```bash
env -u ALL_PROXY -u all_proxy ...
```

不要重复 Phase 0 runtime tests，除非环境、驱动、pins 或依赖实际改变。

## 5. Phase 1A 固定数据与 contract

```text
repo: lerobot/pusht
revision: 7628202a2180972f291ba1bc6723834921e72c19
local root: /home/kxj/.cache/huggingface/lerobot/lerobot/pusht
payload: 8 files / 7,686,801 bytes / 7.5 MiB du
Dataset: v3.0 / 206 episodes / 25,650 frames / 1 task / 10 Hz
```

固定 validation IDs：

```text
[4, 8, 16, 20, 22, 39, 40, 41, 42, 43, 61, 65, 67, 96, 115, 119, 138, 150, 156, 182, 199]
```

- split：length-stratified、21 strata、seed `20260825`；
- train：185 episodes / 23,043 frames；
- validation：21 episodes / 2,607 frames；
- validation frame fraction：10.1637%；
- train/validation mean episode length：124.5568 / 124.1429；
- chunk50 padded slots：19.6697% / 19.7353%。

数据/ACT contract：

- stored image HWC `[96,96,3]` video；decoded image CHW float32 `[3,96,96]`、range `[0,1]`；
- state/action 都是 float32 `[2]`；action 是 absolute target `[x,y]`；
- VISUAL/STATE/ACTION normalization 使用 snapshot `MEAN_STD`；
- real-stat action inverse roundtrip max error `7.62939453125e-06`；
- episode 尾部 action query clamp 到末 action，并由 `action_is_pad` 标记；VAE attention 与 L1 loss
  都排除 padding；
- `next.success` 全 false，不能作为 rollout success metric；后续应使用 PushT 环境评测协议。

Phase 1A 综合复现入口：

```bash
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
HF_HUB_DISABLE_TELEMETRY=1 \
python scripts/check_phase1a_pusht_dataset.py \
  --config configs/dataset/pusht_v3.yaml
```

该 gate 已 PASS，不要无原因重复。

## 6. Phase 1A/1B 已知预期例外与失败恢复

- Image stats count 19,619：LeRobot 每 episode 最多抽样100帧，复算精确匹配，不是数据缺损。
- TorchCodec 受 FFmpeg/CXXABI loader 链路影响会报告长 warning；显式 PyAV 已验证并固定。不安装或
  修改 TorchCodec/FFmpeg。
- Phase 1A processor 首次 harness 把未 collate `[100,2]` action 错当成 batch，失败后改为
  DataLoader-like batch，retry PASS。
- 第一版 stratified split 的逐行 report 因排序后错误 pairing 而错配；实际集合/metrics 正确，固化
  脚本已修复并 PASS。
- Phase 1B config checker 首次错误假定 stats 一定是 Torch Tensor，对 NumPy array 调用 `.cpu()`；
  改用 `torch.as_tensor(...)` 后严格离线 retry PASS。失败没有留下 output/cache/payload 副作用。

## 7. Phase 1B trainer/config 源码结论

已只读确认：

- trainer defaults：batch 8、steps 100,000、workers 4、spawn、env eval/save 每20,000步；
- gradient accumulation default 1，effective batch = batch × data-parallel workers × accumulation；
- Accelerator default precision 是 `no`；training BF16 由 `accelerator.mixed_precision=bf16` 与
  `accelerator.autocast()` 控制；
- ACT default chunk/action steps 是100/100；项目显式覆盖为50/10；
- ACT default 会加载 ImageNet ResNet18 weights；项目显式设置
  `pretrained_backbone_weights=null`；
- ACT optimizer preset 是 AdamW：lr `1e-5`、weight decay `1e-4`、betas `(0.9,0.999)`、eps
  `1e-8`、grad clip norm `10.0`，scheduler `None`；
- `DatasetConfig.use_imagenet_stats` default `True` 会覆盖 camera stats；项目必须显式 false；
- `PreTrainedConfig.push_to_hub` default `True`；项目必须显式
  `policy.push_to_hub=false` 和 `save_checkpoint_to_hub=false`；
- `eval_split` 按 task 取 episode list 尾部。项目 training config 将185个 train IDs 放前面、21个固定
  validation IDs 放末尾，再用 `eval_split=21/206` 精确复现 Phase 1A split；离线 factory 已断言。

## 8. Phase 1B 保存的复现资产

- `configs/train/act_pusht_phase1b_smoke.yaml`：已提交的2-step BF16 smoke config；
- `scripts/check_phase1b_act_config.py`：已提交的严格离线 config/dataset/safety audit；
- `results/phase1b_act_pusht_smoke_2step.yaml`：已提交的小型结构化结果；
- `results/raw/phase1b_act_pusht_smoke_2step.log`：完整原始日志，Git ignored；
- `experiments/act_baseline/README.md`：完整 gate、例外、结果与解释；
- `docs/phases/phase_01a_pusht_dataset.md`：Phase 1A 完成记录。

原始日志：

```text
bytes: 22,761
lines: 511
sha256: e31ffd3e9c7de62fc336827069969e2e2043bcd55a0b99564c3e7f6d5c4251bc
```

不要把大段 raw log 纳入 Git；结构化摘要和关键解释已经提交。

## 9. Phase 1B config/dataset preflight PASS

严格离线 checker 最终输出：

```text
PHASE 1B ACT CONFIG/DATASET PREFLIGHT: PASS
Dataset payload remained unchanged and the temporary datasets cache was removed.
No model, CUDA context, training, checkpoint, network, W&B, GUI, or upload was used.
```

确认：

- 2 steps / batch 1 / accumulation 1 / BF16；
- chunk 50 / execute 10；
- 185/21 split 精确匹配；
- fixed SHA / PyAV / snapshot image stats；
- W&B、policy push、checkpoint push、env eval、eval recording、checkpoint writing、pretrained download
  全部关闭。

## 10. 2-step BF16 training smoke PASS

训练由用户单独授权，严格离线运行：

```text
exit status: 0
steps: 2
effective batch: 1
precision: BF16
learnable parameters: 51,563,394
optimizer: AdamW, lr 1e-5, grad clip 10.0
scheduler: None
```

逐步结果：

| Step | Loss | L1 | KL | Pre-clip grad norm | Step time | Peak PyTorch allocation |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 77.405 | 0.767 | 7.664 | 1792.646 | 1.067 s | 0.86 GiB |
| 2 | 70.916 | 0.621 | 7.029 | 1696.500 | 0.044 s | 0.94 GiB |

进程统计：

```text
whole-process wall: 7.24 s
user time: 13.80 s
system time: 3.43 s
max RSS: 2,614,364 KiB
socket messages sent/received: 0 / 0
```

边界：

- grad norm 是 `accelerator.clip_grad_norm_` 返回的 clipping 前 norm，实际 threshold 是10.0；值有限、
  optimizer step 成功，但较大，overfit 时继续观察；
- 两个不同 sample 的 loss 下降不能证明 convergence/generalization；
- step 2 的44 ms 只是一个 warm-up 后 step，不是正式吞吐 benchmark；
- trainer 报告0.94 GiB 是本轮 PyTorch peak allocation；进程退出后的 `nvidia-smi` 数值不是训练峰值；
- `save_checkpoint=false`，没有创建 checkpoint 或 output directory；
- 没有 W&B、Hub push、pretrained download、env eval、GUI 或网络 socket；
- TorchCodec warning 后 fallback PyAV 为预期例外。

## 11. 当前尚未完成

- 选择 overfit 目标：小 episode 子集还是完整 train split 的短程 loss sanity run；
- 决定 overfit steps、batch size、gradient accumulation、eval-loss cadence 和 stop criteria；
- 决定是否在 overfit gate 写 checkpoint，以及 checkpoint cadence/manifest；
- 选择本地结构化 metric sink（至少 YAML/JSONL；是否接 TensorBoard 需先只读核对 trainer）；
- 获得新的训练授权后运行 overfit；
- 分析 loss/grad/VRAM/wall time；
- 生成可加载 checkpoint；
- 固定 PushT rollout success protocol 并运行评测；
- 保存成功/失败 rollout、success rate、推理 latency/FPS；
- Phase 1B 完成后再进入 Diffusion Policy 对照与 LIBERO。

## 12. 下一阶段只读计划

1. 完整读取本 handoff、ACT README、smoke config/result 和 Phase 1A dataset config。
2. 只读核对 Git、最近提交、submodule pins 和 raw log hash；不要重复已完成 runtime gates。
3. 只读核对 trainer 的本地 metric/checkpoint 行为与 eval-loss consumption。
4. 基于23043 train frames、2607 validation frames、0.94 GiB batch-1 smoke peak，形成一个明确的
   overfit config 候选和 go/no-go/停止标准，但不运行。
5. 保存 config 与离线 checker，宿主审计 PASS 后更新记录。
6. 单独向用户请求 overfit 训练授权。
7. 每次用户粘贴宿主输出：先 PASS/FAIL/预期例外，解释边界，更新记录，再给下一条最小安全命令。

## 13. 新对话恢复提示模板

新对话应让助手先完整读取本文件，并明确以下要求：

```text
请继续 Embodied VLA Lab 项目。

项目根目录：
/home/kxj/embodied_ws/embodied_vla_lab

请先完整读取最新 handoff：
/home/kxj/embodied_ws/embodied_vla_lab/docs/handoffs/2026-08-25_phase1b_smoke_complete_overfit_handoff.md

然后完整读取：
1. experiments/act_baseline/README.md
2. configs/train/act_pusht_phase1b_smoke.yaml
3. results/phase1b_act_pusht_smoke_2step.yaml
4. configs/dataset/pusht_v3.yaml
5. docs/phases/phase_01a_pusht_dataset.md

再只读核对 git status、最近提交、三个 submodule pins，以及 raw smoke log 的 SHA-256。

Phase 0、Phase 1A、Phase 1B config preflight 和2-step BF16 training smoke 都已经完成，不要重复。
最新 handoff 是未提交的本地文件，不要视为异常，不要未经授权 commit。

下一步先只读设计 overfit config 与停止标准，不要训练。安装、dataset/model 下载、训练、GUI、
W&B/login/upload、commit、push 都必须分别获得我的明确授权。不要使用多 agent。

之后我每发一段宿主运行结果，你都需要：
- 先判断 PASS、FAIL 或预期例外；
- 解释结果、数值意义与边界；
- 更新项目记录和可复现资产；
- 再给下一条最小、安全的宿主命令。
```
