# Upstream Dependency and Code Analysis

更新时间：2026-08-25。

本分析基于项目 submodule 中实际固定的代码，而不是只参考在线 README。

## 1. 固定版本

| Repository | Commit | Commit date | Branch at clone |
|---|---|---|---|
| LeRobot | `8b256a6c0d4769c3cc3e7e98f04940126398a391` | 2026-08-24 | `main` |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | 2025-03-15 | `master` |
| OpenVLA | `c8f03f48af692657d3060c19588038c7220e9af9` | 2025-03-23 | `main` |

三个仓库均采用 shallow submodule。父仓库记录精确 gitlink，因此后续更新不会自动改变实验版本。

## 2. LeRobot

### 当前依赖事实

固定版本为 LeRobot `0.6.2`：

| Component | Constraint |
|---|---|
| Python | `>=3.12` |
| PyTorch | `>=2.7,<2.12.0` |
| torchvision | `>=0.22.0,<0.27.0` |
| NumPy | `>=2.0,<2.3` |
| Transformers extra | `>=5.4,<5.6` |
| Diffusers extra | `>=0.38,<0.40` |
| W&B | `>=0.24,<0.28` |
| CUDA index in source config | PyTorch cu128 |

源码安装文档明确说明 cu128 driver floor 为 570.86，并覆盖 Blackwell。宿主驱动为 580.173.02，满足该约束。

### 可选 extras

- `training`：dataset、accelerate、W&B。
- `pusht`：dataset、gym-pusht、pymunk。
- `diffusion`：diffusers。
- `libero`：dataset、Transformers、SciPy、`hf-libero>=0.1.4,<0.2.0`。

不使用 `.[all]`，因为它会安装大量无关机器人硬件、VLA、仿真和开发依赖，增加 resolver 冲突和磁盘占用。

### 分阶段安装策略

1. Phase 0/1A 只安装 `training,pusht`。
2. PushT ACT smoke test 通过后添加 `diffusion`。
3. Phase 2 前再添加 `libero`。

这样能区分 LeRobot core、Diffusion Policy 和 LIBERO 三层问题。

### 关键代码入口

#### Dataset

- `src/lerobot/datasets/lerobot_dataset.py`：主数据集接口。
- `src/lerobot/datasets/dataset_metadata.py`：metadata 与统计信息。
- `src/lerobot/datasets/feature_utils.py`：feature shape/type 工具。
- `src/lerobot/datasets/streaming_dataset.py`：流式数据读取。
- `src/lerobot/datasets/video_utils.py`：视频后端。

#### ACT

- `src/lerobot/policies/act/configuration_act.py`：输入 feature、chunk、normalization 和模型超参数。
- `src/lerobot/policies/act/processor_act.py`：策略前后处理。
- `src/lerobot/policies/act/modeling_act.py`：`ACTPolicy`、`select_action`、`predict_action_chunk`、VAE、encoder/decoder 和 temporal ensembler。

ACT 默认对 action 使用 mean/std normalization。训练损失在 `ACTPolicy.forward` 中组合 action L1 与 VAE KL 项；必须结合 config 阅读，不能只看最终 loss。

#### Diffusion Policy

- `src/lerobot/policies/diffusion/configuration_diffusion.py`
- `src/lerobot/policies/diffusion/processor_diffusion.py`
- `src/lerobot/policies/diffusion/modeling_diffusion.py`

#### Training and evaluation

- `src/lerobot/scripts/lerobot_train.py`
- `src/lerobot/scripts/lerobot_eval.py`
- `src/lerobot/configs/train.py`
- `src/lerobot/utils/train_utils.py`

#### LIBERO wrapper

- `src/lerobot/envs/configs.py`：`LiberoEnv` 配置，默认 20 Hz。
- `src/lerobot/envs/libero.py`：suite/task 构建、相机映射、state/action space、reset 和 step。
- `src/lerobot/envs/factory.py`：统一环境工厂。

### LIBERO 数据契约（LeRobot 当前文档）

- 两个 256×256 RGB 相机：agent view 和 eye-in-hand。
- state dimension：8。
- action dimension：7。
- control frequency：20 Hz。
- 支持 relative 和 absolute control mode，必须与 checkpoint 一致。

### 日志结论

当前源码原生集成 W&B，但全仓库未发现 `torch.utils.tensorboard.SummaryWriter` 或 TensorBoard 写入路径。因此：

- Phase 1 以 W&B 和本地结构化日志为主。
- TensorBoard 不在首次安装中强行加入训练器。
- ACT baseline 跑通后，再决定实现最小 callback 或离线日志转换脚本。

## 3. Original LIBERO

### 上游依赖事实

原始 README 使用：

- Python 3.8.13。
- PyTorch 1.11.0 + CUDA 11.3。
- NumPy 1.22.4。
- Transformers 4.21.1。
- robosuite 1.4.0、gym 0.25.2。

这套依赖与 Blackwell、LeRobot Python 3.12、NumPy 2 和 PyTorch cu128 不兼容。

### 决策

- 不执行 `libero/requirements.txt`。
- 不在 `evla-core` 中 editable-install 原始 LIBERO。
- 原始 submodule 用于阅读 benchmark、BDDL、环境、数据和评测语义。
- 实际 Phase 2 runtime 使用 LeRobot `libero` extra 提供的 `hf-libero`。
- smoke test 时核对 `hf-libero` 的 task、action、camera 和 success semantics 与原始仓库一致。

### 关键代码入口

- `libero/libero/benchmark/`：suite、task map 和固定初始化状态。
- `libero/libero/envs/bddl_base_domain.py`：任务、机器人、传感器与 20 Hz 控制环境。
- `libero/libero/envs/env_wrapper.py`：`OffScreenRenderEnv`。
- `libero/lifelong/datasets.py`：原始 HDF5/robomimic sequence dataset wrapper。
- `libero/lifelong/main.py`：训练入口。
- `libero/lifelong/metric.py`：success evaluation。
- `libero/lifelong/evaluate.py`：独立评测入口。

原始最小示例使用 7 维 dummy action 和 sparse success reward。这将作为 LeRobot wrapper smoke test 的语义对照。

## 4. OpenVLA

### 上游依赖事实

OpenVLA `0.0.3` 的 `pyproject.toml` 固定：

| Component | Pin |
|---|---|
| Python | `>=3.8`，官方开发环境 3.10 |
| PyTorch | `==2.2.0` |
| torchvision | `==0.17.0` |
| torchaudio | `==2.2.0` |
| Transformers | `==4.40.1` |
| tokenizers | `==0.19.1` |
| timm | `==0.9.10` |
| PEFT | `==0.11.1` |
| FlashAttention | README 指定 `2.5.5` |

完整 editable install 还会安装 TensorFlow、TFDS、tensorflow-graphics 和 Git 依赖 `dlimp`。这些主要服务 RLDS/训练，并非最小推理都需要。

### 与本机的冲突

- PyTorch 2.2 官方组合早于 Blackwell 支持，不能作为 RTX 5070 Ti runtime。
- OpenVLA Transformers 4.40 与 LeRobot Transformers 5.4–5.5 互斥。
- FlashAttention 2.5.5 不是本机第一轮应编译的目标。
- 因此 OpenVLA 必须使用独立 `evla-openvla` 环境。

### Phase 3 安装决策

- 不直接执行 `python -m pip install -e openvla`，否则会尝试固定 PyTorch 2.2。
- 先创建 curated minimal-inference requirements。
- 显式安装支持 Blackwell 的 cu128 PyTorch。
- 保留 Transformers/tokenizers/timm 的 OpenVLA 兼容版本，逐项 smoke test。
- local package 如需 editable install，使用 `--no-deps`，依赖由我们显式控制。
- attention 首先使用 eager/SDPA；不安装 FlashAttention 2.5.5。

### 显存判断

OpenVLA 自带 verification script 注释给出：

- 8-bit：约 9 GB passive / 10 GB active。
- 4-bit：约 6 GB passive / 7 GB active。

对 12 GB Laptop GPU，4-bit 是首选；8-bit 的剩余空间过小。该数字只是上游脚本说明，必须在本机实测峰值显存。

官方普通 LoRA 建议至少约 27 GB VRAM，不适合本机。`vla-scripts/finetune.py` 支持 `use_quantization=True` 的 4-bit LoRA，但 12 GB 上能否完成 backward 尚未验证，因此归类为 go/no-go 实验。

### 关键代码入口

- `prismatic/`：VLM/VLA backbone、数据和训练基础设施。
- `prismatic/vla/action_tokenizer.py`：动作 tokenization。
- `prismatic/models/vlas/openvla.py`：动作预测和反归一化。
- `vla-scripts/extern/verify_openvla.py`：BF16/8-bit/4-bit 推理样例。
- `vla-scripts/finetune.py`：LoRA/4-bit LoRA。
- `experiments/robot/libero/run_libero_eval.py`：LIBERO rollout 与 4-bit flag。

## 5. 最终环境边界

```text
evla-core
  Python 3.12
  PyTorch 2.11 + cu128 candidate
  LeRobot 0.6.2
  Phase 1: training + pusht
  Phase 1B: + diffusion
  Phase 2: + libero (hf-libero)

evla-openvla
  Python 3.10
  Blackwell-compatible cu128 PyTorch
  OpenVLA-compatible Transformers 4.40 stack
  4-bit inference first
  no FlashAttention first
```

选择 PyTorch 2.11 作为 `evla-core` 候选，因为它是 LeRobot 当前 `<2.12` 范围内的最新受支持版本，并有 cu128 wheel。只有 resolver dry-run 和实际 CUDA smoke test 通过后才写入最终 lock。

## 6. 下一步

1. 检查 submodule 与父仓库 staged 状态。
2. 不安装的前提下准备 `evla-core` 安装计划和验证命令。
3. 用户正常退出 Unitree GPU 任务后，创建 `evla-core`。
4. 先安装 Python/FFmpeg/PyTorch，再验证 GPU。
5. GPU gate 通过后安装 LeRobot `training,pusht`。
