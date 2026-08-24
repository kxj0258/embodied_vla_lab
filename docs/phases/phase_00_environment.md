# Phase 0 — Environment Preparation

## 实验目的

为 RTX 5070 Ti Laptop GPU 建立可复现、相互隔离的机器人学习环境。在引入 MuJoCo、LIBERO 和大模型前，先验证驱动、PyTorch、Blackwell CUDA 架构、LeRobot、PushT 和 ACT 的最小接口。

## 当前状态

状态：进行中。

### 已完成

- [x] 收集宿主机 GPU、驱动、OS、conda 和工具链信息。
- [x] 固定 LeRobot、LIBERO、OpenVLA 三个上游 submodule commit。
- [x] 分析三个仓库的依赖边界与关键代码入口。
- [x] 创建 `evla-core`，安装 Python、FFmpeg 和 PyTorch cu128。
- [x] 验证 RTX 5070 Ti、compute capability 12.0、BF16 和 `sm_120`。
- [x] 安装 LeRobot 0.6.2 editable training 依赖。
- [x] 建立无 GUI 的 PushT reset/step/render 闭环。
- [x] 验证 LeRobot observation preprocessing 与 ACT image-to-action 接口。
- [x] 记录 OpenCV metadata 例外和 PushT 图像尺寸差异。

### 尚未完成

- [ ] 安装和验证 MuJoCo EGL 离屏渲染。
- [ ] 安装 LeRobot 维护的 LIBERO compatibility runtime。
- [ ] 完成 LIBERO reset、dummy step 和离屏 RGB 检查。
- [ ] 决定并验证本地 TensorBoard 记录方案。
- [x] 由宿主终端执行固化后的 `scripts/check_phase0_core.py`。

## 宿主环境

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-138-generic |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| VRAM | 12227 MiB |
| Compute capability | 12.0 / `sm_120` |
| Driver | 580.173.02 |
| Driver CUDA ceiling | 13.0，不等于本地 Toolkit |
| RAM | 32 GB（用户提供） |
| Conda | 26.5.3 via Miniforge |
| Session | X11 |
| Free disk at initial audit | approximately 126 GB |

Codex 受限环境看不到 `/dev/nvidia*`；所有 GPU 验证均由用户在宿主终端执行并反馈原始输出。

## 固定上游版本

| Repository | Commit | Role |
|---|---|---|
| LeRobot | `8b256a6c0d4769c3cc3e7e98f04940126398a391` | dataset、policy、processor、train/eval |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | benchmark 语义与源码分析 |
| OpenVLA | `c8f03f48af692657d3060c19588038c7220e9af9` | VLA inference 与微调源码分析 |

详细依赖分析见 `docs/dependency_analysis.md`。

## evla-core 已验证版本

| Component | Verified version |
|---|---|
| Python | 3.12.14 |
| pip | 26.2.1 |
| FFmpeg | 7.1.1, conda-forge |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| LeRobot | 0.6.2, editable pinned submodule |
| gymnasium | 1.3.0 |
| gym-pusht | 0.1.6 |
| OpenCV | `opencv-python-headless` 4.13.0.92 |
| datasets | 4.8.5 |
| av | 15.1.0 |
| torchcodec | 0.11.1 |
| accelerate | 1.14.0 |
| W&B | 0.27.2 |

环境基础层记录在 `configs/env/evla-core.yml`。该文件不伪装成完整一键安装文件：PyTorch cu128 自定义索引、LeRobot editable 安装以及 PushT OpenCV workaround 仍按下面的分阶段命令执行。

## 执行与结果

### D1：conda 基础层

验证命令：

```bash
conda activate evla-core
which python
python --version
python -m pip --version
which ffmpeg
ffmpeg -version
```

结果：Python 3.12.14、pip 26.2.1、FFmpeg 7.1.1；环境初始大小约 750 MB。

### D2：PyTorch cu128

安装命令：

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

结果：

- `torch==2.11.0+cu128`。
- `torchvision==0.26.0+cu128`。
- `torch.version.cuda == "12.8"`。
- 安装后 `pip check` 无损坏依赖。
- 环境大小约 7.4 GB。

### D3：CUDA / Blackwell / BF16

实测结果：

| Check | Result |
|---|---|
| CUDA available | True |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Compute capability | `(12, 0)` |
| PyTorch arch list | includes `sm_120` |
| BF16 | supported |
| 4096×4096 BF16 matmul | finite, PASS |
| Matmul time | 0.103391 s，仅 smoke test |
| Peak PyTorch allocation | 184.12 MiB |

测试前空闲显存约 10.7 GB；测试结束后显存回落到桌面基线。该结果验证运行时兼容性，不作为模型训练吞吐指标。

### D4A：LeRobot resolver dry-run

直接解析 `.[training,pusht]` 会同时引入：

- LeRobot core 依赖的 `opencv-python-headless`。
- `gym-pusht` metadata 依赖的 `opencv-python`。

两者导出同一个 `cv2` 命名空间，因此没有直接接受该组合。

### D4B：LeRobot training 与 PushT runtime

LeRobot 安装：

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab/lerobot
python -m pip install -e ".[training]"
```

PushT 依赖采用分阶段安装：

```bash
python -m pip install \
  pygame==2.6.1 \
  pymunk==6.11.1 \
  shapely==2.1.2 \
  scikit-image==0.26.0

python -m pip install --no-deps gym-pusht==0.1.6
```

结果：

- 仅安装 `opencv-python-headless==4.13.0.92`，未安装 `opencv-python`。
- PushT reset、10 个随机 step 和 `rgb_array` render 均通过。
- 观测 keys：`agent_pos`、`pixels`。
- 图像观测：`uint8[96,96,3]`。
- state：`float64[2]`，LeRobot preprocessing 后为 `float32[1,2]`。
- action space：二维连续绝对目标位置，范围 `[0,512]`。
- visualization frame：`uint8[384,384,3]`。

`pip check` 的唯一结果为：

```text
gym-pusht 0.1.6 requires opencv-python, which is not installed.
```

这是已知 metadata 例外，不是缺少 `cv2`：实际 `cv2` import、物理步进和离屏渲染均已通过。禁止为了消除此提示而再安装非 headless OpenCV。

### D4C：LeRobot wrapper 与 ACT 接口

LeRobot observation preprocessing 已验证：

```text
pixels: uint8[B,H,W,C]
  → observation.image: float32[B,C,H,W], range [0,1]

agent_pos: float64[B,2]
  → observation.state: float32[B,2]
```

当前 pinned LeRobot 的默认 `PushtEnv` 声明 `observation.image=(3,384,384)`，但 `gym_kwargs` 未传递 observation width/height，`gym-pusht==0.1.6` 实际输出 `(3,96,96)`。本项目通过 `configs/env/pusht_96.yaml` 显式把 policy contract 固定为 96×96；暂不修改上游 submodule。

随机初始化 ACT 接口结果：

| Item | Result |
|---|---|
| Input image | `float32[1,3,96,96]` |
| Input state | `float32[1,2]` |
| Output action chunk | `float32[1,100,2]` |
| `chunk_size` | 100 |
| `n_action_steps` | 10 |
| Backbone | ResNet18, no pretrained download |
| Parameters | 51.59 M |
| First-pass latency | 0.309598 s |
| Peak PyTorch allocation | 222.56 MiB |
| Result | PASS |

延迟包含首次 CUDA/model warm-up；显存仅是随机初始化 FP32 inference。二者均不得当作训练性能或正式 benchmark。随机动作没有控制意义，未送入环境。

### D5：固化脚本宿主复核

2026-08-25 在宿主终端执行 `python scripts/check_phase0_core.py`，四个 gate 全部通过：

| Gate | Result |
|---|---|
| package contract | PASS |
| CUDA / BF16 | PASS |
| LeRobot PushT wrapper | PASS |
| ACT image-to-action | PASS |
| First-pass latency | 0.282970 s，非正式 benchmark |
| Peak PyTorch allocation | 223.43 MiB |

复核前 `nvidia-smi` 显示 GPU 已使用 4208 MiB、空闲 7540 MiB、利用率 58%。脚本仍通过，说明该检查所需显存余量较小；由于 GPU 非空闲，延迟不用于横向比较。

## 固化文件

- `configs/env/evla-core.yml`：conda 管理的最小基础层。
- `configs/env/pusht_96.yaml`：PushT 96×96 observation/action runtime contract。
- `scripts/check_phase0_core.py`：宿主侧可重复执行的 core smoke test。

## 下一步

1. 只读分析 LeRobot `libero` extra 的当前 resolver。
2. 明确 MuJoCo、`hf-libero` 与 Python 3.12 的安装集合。
3. 经单独授权后安装最小 MuJoCo/LIBERO runtime。
4. 使用 EGL 完成 reset、dummy step、RGB render，不启动 GUI。

## 思考

- 驱动上限、PyTorch wheel runtime 和 compute capability 必须分开记录；`nvidia-smi` 的 CUDA 13.0 不意味着应安装 CUDA 13 Toolkit。
- 环境 feature declaration 必须与 simulator 实际 tensor 对照。只读取配置会漏掉当前 PushT 384/96 的不一致。
- 包管理器的 metadata 完整性和运行时接口完整性不是同一件事。OpenCV workaround 必须保留原因、实测证据和禁止操作。
- ACT 的 action chunk shape 只说明接口正确；没有 dataset statistics、训练权重和 postprocessor 时，动作数值不可解释。
