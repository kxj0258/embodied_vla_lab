# Compatibility Matrix

更新时间：2026-08-25。

本文件区分“宿主机事实”“已通过运行时验证”和“后续计划”。

## 宿主机

| 项目 | 结果 | 状态 |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS | 已验证 |
| Kernel | 6.8.0-138-generic | 已验证 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 已验证 |
| VRAM | 12227 MiB | 已验证 |
| GPU architecture | Blackwell, compute capability 12.0 | 已验证 |
| NVIDIA driver | 580.173.02 | 已验证 |
| Driver CUDA ceiling | 13.0 | 已验证，不等于 Toolkit |
| RAM | 32 GB（用户提供） | 待命令复核 |
| Free disk at initial audit | 约 126 GB | 已验证 |
| Desktop session | X11 | 已验证 |
| conda | Miniforge, conda 26.5.3 | 已验证 |
| nvcc / system CUDA Toolkit | 未安装 | 当前不阻塞 |

Codex 沙箱不可访问宿主 GPU，CUDA 命令由用户在宿主终端执行。GPU 工作前重新检查其他进程占用，不把关闭 Unitree 程序作为永久环境要求。

## 关键解释

`nvidia-smi` 的 CUDA 13.0 是驱动可支持的最高 CUDA runtime，不表示系统安装了 CUDA 13 Toolkit。当前预编译 PyTorch wheel 自带 CUDA 12.8 runtime；在必须编译 CUDA extension 前不安装系统 Toolkit。

## 固定上游版本

| Repository | Commit |
|---|---|
| LeRobot | `8b256a6c0d4769c3cc3e7e98f04940126398a391` |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| OpenVLA | `c8f03f48af692657d3060c19588038c7220e9af9` |

## Environment A：evla-core

用途：LeRobot、PushT、ACT、Diffusion Policy、LIBERO、MuJoCo、ego-view 和小型 world model。

| Component | Locked/verified value | Status |
|---|---|---|
| Python | 3.12.14 | 已验证 |
| pip | 26.2.1 | 已验证 |
| FFmpeg | 7.1.1, conda-forge | 已验证 |
| PyTorch | 2.11.0+cu128 | CUDA/BF16 已验证 |
| torchvision | 0.26.0+cu128 | ACT forward 已验证 |
| CUDA architecture | `(12,0)`, `sm_120` | 已验证 |
| LeRobot | 0.6.2 editable pinned source | 已验证 |
| gym-pusht | 0.1.6 | headless reset/step/render 已验证 |
| OpenCV | headless 4.13.0.92 only | 已验证 |
| W&B | 0.27.2 | import 已验证，login/run 待验证 |
| TensorBoard | 未安装 | 记录方案待决定 |
| MuJoCo EGL | 未安装/未验证 | Phase 0 下一 gate |
| LIBERO runtime | 未安装/未验证 | Phase 0 下一 gate |

原始 LIBERO 的 Python 3.8、PyTorch 1.11、CUDA 11.3 requirements 不进入该环境。实际 runtime 计划使用 LeRobot 当前 `libero` extra 的 `hf-libero` compatibility package，安装前重新 dry-run。

### 已知例外：OpenCV package metadata

LeRobot core 使用 `opencv-python-headless`，`gym-pusht==0.1.6` metadata 则要求 `opencv-python`。为了避免同一个 `cv2` namespace 的两个 wheel 共存，PushT 使用 `--no-deps` 安装并保留 headless wheel。

因此当前 `pip check` 有且仅有下面一条已接受例外：

```text
gym-pusht 0.1.6 requires opencv-python, which is not installed.
```

如果出现其他 `pip check` 错误，不能归入该例外。

### 已知例外：PushT feature declaration

Pinned LeRobot `PushtEnv` 默认声明 `(3,384,384)` image feature，但其 `gym_kwargs` 没有向 `gym-pusht==0.1.6` 传 observation width/height，实际图像为 `(3,96,96)`。项目基线使用 `configs/env/pusht_96.yaml` 显式对齐声明和 runtime，不修改上游 submodule。

## Environment B：evla-openvla

到 Phase 3 再创建。

| Component | Candidate | Status |
|---|---|---|
| Python | 3.10 | 计划 |
| PyTorch | Blackwell-compatible cu128 build | 待验证 |
| Quantization | 4-bit bitsandbytes | 计划 |
| Batch size | 1 | 计划 |
| Attention | eager/SDPA first | 计划 |
| FlashAttention | 第一轮不安装 | 已决定 |

Pinned OpenVLA 将 PyTorch 固定为 2.2.0、Transformers 固定为 4.40.1，并建议 FlashAttention 2.5.5。该组合不能原样用于 Blackwell，也不能与 LeRobot 的新依赖栈共存。Phase 3 使用独立 curated minimal environment，不直接执行普通 editable install。

## Resource budget

- 单 GPU，12 GB VRAM；训练优先 BF16、小 batch 和梯度累积。
- OpenVLA 优先 4-bit inference；QLoRA 先做显存 go/no-go。
- conda 环境、数据、checkpoint 和视频均需控制磁盘预算。
- 一次只保留当前阶段需要的数据子集和大型 checkpoint。
- 权重、数据和大视频不进入 Git。

## 待验证清单

- [ ] `free -h` 复核 RAM 与 swap。
- [x] 空闲状态可用显存约 10.7 GB（一次实测，任务前仍需复查）。
- [x] PyTorch cu128 识别 compute capability 12.0。
- [x] BF16 tensor 运算。
- [x] LeRobot import、PushT reset/step/render 和 ACT forward。
- [x] 固化的 Phase 0 core script 在宿主终端完整复核。
- [ ] MuJoCo EGL renderer。
- [ ] LIBERO reset/step/离屏图像。
- [ ] W&B login、offline/online run。
- [x] LeRobot 当前源码未发现原生 TensorBoard SummaryWriter。
- [ ] 决定 TensorBoard callback 或离线转换方案。

## 上游参考

- LeRobot: <https://github.com/huggingface/lerobot>
- LIBERO: <https://github.com/Lifelong-Robot-Learning/LIBERO>
- OpenVLA: <https://github.com/openvla/openvla>
- PyTorch Blackwell support: <https://pytorch.org/blog/pytorch-2-7/>
- NVIDIA CUDA architecture matrix: <https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html>
