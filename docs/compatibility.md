# Compatibility Matrix

更新时间：2026-08-25。

本文件区分“宿主机已验证事实”“上游约束”和“计划版本”。计划版本只有通过 smoke test 后才会锁定。

## 宿主机已验证

| 项目 | 结果 | 状态 |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS | 已验证 |
| Kernel | 6.8.0-138-generic | 已验证 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 已验证 |
| VRAM | 12227 MiB | 已验证 |
| GPU architecture | Blackwell, compute capability 12.0 | 已验证 |
| NVIDIA driver | 580.173.02 | 已验证 |
| Driver CUDA ceiling | 13.0 | 已验证，不等于本地 Toolkit |
| RAM | 32 GB（用户提供） | 待命令复核 |
| Free disk | 约 126 GB | 已验证 |
| Desktop session | X11 | 已验证 |
| conda | Miniforge, conda 26.5.3 | 已验证 |
| GCC | 11.4.0 | 已验证 |
| CMake | system 3.22.1 | 已验证 |
| FFmpeg | 未安装 | 已验证 |
| nvcc / CUDA Toolkit | 未安装 | 已验证，当前不阻塞 |

检查时 `unitree_rl` Python 进程占用约 4.3 GB VRAM。文档阶段可保持运行；GPU smoke test、训练和 OpenVLA 加载前需要正常关闭并重新确认空闲显存。

Codex 受限执行环境看不到 `/dev/nvidia*`，但宿主普通终端可以正常访问 GPU。因此 CUDA 相关命令由用户在宿主终端执行并反馈结果。

## 关键解释

`nvidia-smi` 显示的 CUDA 13.0 是当前驱动可支持的最高 CUDA runtime 版本，不表示系统已安装 CUDA 13 Toolkit。预编译 PyTorch wheel 自带所需 CUDA runtime；当前阶段不安装 `nvcc`。

## 固定上游版本

| Repository | Commit |
|---|---|
| LeRobot | `8b256a6c0d4769c3cc3e7e98f04940126398a391` |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| OpenVLA | `c8f03f48af692657d3060c19588038c7220e9af9` |

详细依赖与代码入口见 `docs/dependency_analysis.md`。

## Environment A：evla-core

用途：LeRobot、PushT、ACT、Diffusion Policy、LIBERO、MuJoCo、ego-view、小型 world model。

| 组件 | 候选 | 决策状态 |
|---|---|---|
| Python | 3.12 | 待创建 |
| PyTorch | 2.11 + cu128 candidate | resolver 与 CUDA smoke test 后锁定 |
| CUDA wheel | cu128 | 已选方向 |
| LeRobot | commit `8b256a6c...`，version 0.6.2 | 已固定 |
| LIBERO runtime | LeRobot `hf-libero` compatibility package | 待验证 |
| MuJoCo rendering | EGL offscreen | 待验证 |
| FFmpeg | conda environment package | 待安装 |

固定 LeRobot 要求 Python `>=3.12`、PyTorch `>=2.7,<2.12`，并将 Linux torch 源指向 cu128。选择 PyTorch 2.11 candidate 是为了使用该范围内最新受支持组合；最终以 resolver 和实机 smoke test 为准。

原始 LIBERO 的 Python 3.8/PyTorch 1.11/CUDA 11.3 requirements 不进入此环境。实际 runtime 使用 LeRobot `libero` extra 的 `hf-libero`。

## Environment B：evla-openvla

到 Phase 3 再创建。

| 组件 | 候选 | 决策状态 |
|---|---|---|
| Python | 3.10 | 计划 |
| PyTorch | Blackwell-compatible cu128 build | 待兼容测试 |
| Quantization | 4-bit bitsandbytes | 计划 |
| Batch size | 1 | 计划 |
| Attention | eager/SDPA first | 计划 |
| FlashAttention | 第一轮不安装 | 已决定 |

固定 OpenVLA 将 PyTorch 固定为 2.2.0、Transformers 固定为 4.40.1，并建议 FlashAttention 2.5.5。不能在 Blackwell 上原样照搬，也不能与 LeRobot Transformers 5.4–5.5 共存。Phase 3 使用 curated minimal environment，不直接执行普通 editable install。

## Disk budget

工作盘当前剩余约 126 GB。建议预算：

- conda 环境与包缓存：20–30 GB。
- 三个源码仓库：小于模型和数据成本。
- PushT 数据及产物：控制在数 GB。
- LIBERO 只下载当前阶段所需子集。
- OpenVLA 一次只保留一个目标 checkpoint。
- 模型、数据和视频不进入 Git。

## 待验证清单

- [ ] `free -h` 复核 RAM 与 swap。
- [ ] 空闲状态下 GPU 可用显存。
- [ ] PyTorch cu128 能识别 compute capability 12.0。
- [ ] BF16 tensor 运算。
- [ ] MuJoCo EGL renderer。
- [ ] LeRobot import 和最小 dataset sample。
- [ ] LIBERO reset/step/离屏图像。
- [ ] W&B 本地与在线 logging。
- [x] 检查 LeRobot 当前版本的 TensorBoard 支持方式：未发现原生 SummaryWriter。
- [ ] 决定 TensorBoard callback 或离线转换方案。

## 上游参考

- LeRobot: <https://github.com/huggingface/lerobot>
- LIBERO: <https://github.com/Lifelong-Robot-Learning/LIBERO>
- OpenVLA: <https://github.com/openvla/openvla>
- PyTorch Blackwell support: <https://pytorch.org/blog/pytorch-2-7/>
- NVIDIA CUDA architecture matrix: <https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html>
