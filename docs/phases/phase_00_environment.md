# Phase 0 — Environment Preparation

## 实验目的

为 RTX 5070 Ti Laptop GPU 建立可复现、相互隔离的机器人学习环境，并在安装大型依赖前验证驱动、PyTorch、MuJoCo EGL 和上游仓库的兼容边界。

## 当前状态

状态：进行中。

已完成：

- [x] 收集宿主机 GPU、驱动、OS、conda 和工具链信息。
- [x] 确认 Codex GPU 隔离与宿主驱动故障不是同一问题。
- [x] 初始化本地项目 Git 仓库。
- [x] 建立文档和实验目录骨架。

尚未执行：

- [x] 添加三个上游 submodule，并记录精确 commit。
- [x] 分析三个上游仓库的依赖与关键代码入口。
- [ ] 创建 `evla-core` conda environment。
- [ ] 安装 PyTorch 或任何项目依赖。
- [ ] 下载数据集或 checkpoint。
- [ ] 运行 MuJoCo、训练或 GUI。

## 宿主环境

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-138-generic |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| VRAM | 12227 MiB |
| Compute capability | 12.0 |
| Driver | 580.173.02 |
| Driver CUDA ceiling | 13.0 |
| Conda | 26.5.3 via Miniforge |
| GCC | 11.4.0 |
| System CMake | 3.22.1 |
| FFmpeg | not installed |
| nvcc | not installed |
| Session | X11 |
| Free disk | approximately 126 GB |

## 兼容性判断

- Blackwell 需要使用支持 compute capability 12.0 的 PyTorch build。
- 主环境选择 CUDA 12.8 wheel，避免把驱动上限 CUDA 13.0 误当作本地 Toolkit。
- 不在系统层安装 CUDA Toolkit；只有未来必须编译 CUDA extension 时再单独评估。
- LeRobot/LIBERO 与 OpenVLA 使用不同 conda environment。
- 不复用或修改现有 `unitree_rl` environment。
- FFmpeg 安装在 `evla-core` 内，不使用系统 `/usr/bin/pip`。

## 计划环境

### evla-core

```text
Python 3.12
PyTorch 2.10/2.11 + cu128 (after repository inspection)
LeRobot fixed commit
hf-libero compatibility runtime
MuJoCo EGL
W&B + local reproducibility logs
```

### evla-openvla

Phase 3 创建：

```text
Python 3.10
Blackwell-compatible PyTorch cu128
4-bit quantization
batch size 1
eager/SDPA first
no FlashAttention in first smoke test
```

## 执行顺序

1. ~~添加并固定 LeRobot/LIBERO/OpenVLA submodule。~~
2. ~~只读分析各仓库依赖与关键模块。~~
3. 以 Python 3.12、PyTorch 2.11 + cu128 为 candidate，完成 resolver/安装验证。
4. 创建环境并安装最小依赖。
5. PyTorch CUDA/BF16 smoke test。
6. 安装 LeRobot `training,pusht`。
7. LeRobot import 与 PushT sample smoke test。
8. Phase 2 前再添加 `libero` 并做 EGL reset/step/render。
9. 输出环境锁定文件和结果记录。

## 固定上游 commit

- LeRobot：`8b256a6c0d4769c3cc3e7e98f04940126398a391`
- LIBERO：`8f1084e3132a39270c3a13ebe37270a43ece2a01`
- OpenVLA：`c8f03f48af692657d3060c19588038c7220e9af9`

依赖分析见 `docs/dependency_analysis.md`。

## 验收标准

- [ ] `torch.cuda.is_available()` 为 True。
- [ ] GPU 名称和 capability 正确。
- [ ] BF16 矩阵运算正常。
- [ ] MuJoCo EGL 保存非空 RGB 图像。
- [ ] LeRobot dataset/policy 最小 import 正常。
- [ ] LIBERO 环境完成 reset 和若干 dummy steps。
- [ ] 记录峰值显存、命令和版本。
- [ ] 重新创建环境的步骤经过复核。

## 已知问题

### Codex 中 `nvidia-smi` 失败

宿主终端中 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm` 均存在且 `nvidia-smi` 正常；Codex 受限环境不可见这些设备。因此 GPU 命令由用户在宿主终端运行，并将结果反馈到实验记录。

### Unitree 进程占用 GPU

采集信息时，`unitree_rl` Python 进程占用约 4.3 GB VRAM。文档工作不受影响；开始 GPU smoke test 前通过正常方式退出该程序并重新检查空闲显存。

## 思考

环境复现不是“装成功即可”。必须把 GPU 架构、PyTorch wheel、仿真渲染后端和上游 commit 作为整体锁定，否则后续训练问题可能被误判为算法问题。
