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

## Environment A：evla-core

用途：LeRobot、PushT、ACT、Diffusion Policy、LIBERO、MuJoCo、ego-view、小型 world model。

| 组件 | 候选 | 决策状态 |
|---|---|---|
| Python | 3.12 | 待创建 |
| PyTorch | 2.10 或 2.11 | 克隆并检查 LeRobot commit 后锁定 |
| CUDA wheel | cu128 | 已选方向 |
| LeRobot | fixed Git commit | 待添加 submodule |
| LIBERO runtime | LeRobot `hf-libero` compatibility package | 待验证 |
| MuJoCo rendering | EGL offscreen | 待验证 |
| FFmpeg | conda environment package | 待安装 |

选择 cu128 的原因：支持 Blackwell，同时降低采用 CUDA 13 wheel 对上游机器人学习依赖造成的兼容风险。驱动 580.173.02 满足该方向。

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

OpenVLA 上游历史依赖基于较旧 PyTorch。不能在 Blackwell 上原样照搬，必须将“模型代码兼容性”和“GPU 架构兼容性”分别验证。

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
- [ ] LeRobot 当前版本的 TensorBoard 支持方式。

## 上游参考

- LeRobot: <https://github.com/huggingface/lerobot>
- LIBERO: <https://github.com/Lifelong-Robot-Learning/LIBERO>
- OpenVLA: <https://github.com/openvla/openvla>
- PyTorch Blackwell support: <https://pytorch.org/blog/pytorch-2-7/>
- NVIDIA CUDA architecture matrix: <https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html>
