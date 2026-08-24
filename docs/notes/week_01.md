# Week 01 Notes

周期：2026-08-25 起。

## 本周目标

完成 Phase 0 的核心运行时与 MuJoCo/LIBERO gate：建立可复现的 `evla-core` 环境，理解三个上游仓库的职责和依赖边界。本周不进行正式训练。

## 已完成

- [x] 明确长期项目目标、单 GPU 限制与 simulation-first 路线。
- [x] 分析 LeRobot、LIBERO 和 OpenVLA 的关系。
- [x] 收集宿主机 GPU、驱动、OS、conda、磁盘和工具链信息。
- [x] 初始化项目 Git 仓库、文档骨架和实验目录。
- [x] 以 shallow submodule 固定三个上游仓库及 commit。
- [x] 确定 LeRobot/LIBERO 与 OpenVLA 使用隔离环境。
- [x] 创建 `evla-core`：Python 3.12.14、FFmpeg 7.1.1。
- [x] 安装并验证 PyTorch 2.11.0+cu128 与 torchvision 0.26.0+cu128。
- [x] 验证 Blackwell `sm_120`、BF16 和 CUDA tensor 运算。
- [x] 安装 LeRobot 0.6.2 training 依赖。
- [x] 解决 PushT 的 headless OpenCV 安装冲突。
- [x] 完成 PushT reset/step/render 与 LeRobot preprocessing。
- [x] 完成随机初始化 ACT image/state → action chunk 接口测试。
- [x] 复现并定位 LeRobot PushT 384/96 图像声明差异。
- [x] 新增环境、PushT 配置和可重复 core smoke script。

## 待完成

- [x] 用户在宿主终端复核 `scripts/check_phase0_core.py`。
- [ ] dry-run LeRobot 当前 `libero` extra。
- [ ] 完成 MuJoCo EGL smoke test。
- [ ] 完成 LIBERO reset/step/render。
- [ ] 更新 Phase 0 最终结果并提交环境记录。

## 当前决策

- 主项目语言：中文。
- GitHub 用户：`kxj0258`。
- Hugging Face 用户：`kxj25`。
- 实验追踪：W&B + 本地 TensorBoard/结构化日志。
- 工作节奏：用户在宿主终端执行安装、GPU、下载、训练和 GUI 命令并反馈。
- 可投入时间：每天约 3–4 小时。
- PushT baseline 使用 96×96 image observation，不修改上游 submodule。
- 保留 `opencv-python-headless`，不同时安装 `opencv-python`。
- ACT 首个 baseline 使用 image + 2D agent position 输入和 2D action chunk 输出。

## 实测摘要

| Item | Result |
|---|---|
| GPU / capability | RTX 5070 Ti Laptop / 12.0 |
| PyTorch | 2.11.0+cu128 |
| BF16 matmul | PASS |
| PushT headless reset/step/render | PASS |
| LeRobot processed image | `float32[1,3,96,96]` |
| LeRobot processed state | `float32[1,2]` |
| Random ACT output | `float32[1,100,2]` |
| ACT parameters | 51.59 M |
| ACT first-pass latency | 0.309598 s，非正式 benchmark |
| ACT peak allocation | 222.56 MiB，inference only |
| 固化脚本复核 | 四个 gate 全部 PASS |
| 复核 first-pass / allocation | 0.282970 s / 223.43 MiB，GPU 非空闲 |

## 本周风险与问题

### 1. LeRobot 与 OpenVLA 依赖栈不兼容

必须保持 `evla-core` 和未来 `evla-openvla` 隔离，不能为方便而混装 Transformers/PyTorch 依赖。

### 2. OpenCV metadata 与运行时需求不一致

`gym-pusht` metadata 要求 GUI OpenCV，而 LeRobot 使用 headless wheel。当前通过 `gym-pusht --no-deps` 保留单一 `cv2` provider；`pip check` 的一条提示已被记录为显式例外。

### 3. PushT 声明尺寸与 runtime 不一致

LeRobot 默认声明 384×384，实际 gym observation 为 96×96。当前配置显式锁定 96×96，并将 simulator tensor 作为最终事实来源。

### 4. 12 GB 显存和磁盘预算

ACT/PushT 具备充分余量；OpenVLA 普通 LoRA 不在本机目标内。数据、模型与缓存必须按阶段下载和清理。

### 5. Codex GPU 隔离

Codex 沙箱只能完成源码和文件检查；GPU 证据必须来自宿主终端原始输出。

## 每日记录

### Day 1

- 完成系统信息采集与架构规划。
- 完成本地 Git 仓库和文档基线。
- 固定三个上游 submodule。
- 完成依赖和源码入口分析。

### Day 2

- 创建 `evla-core` 并锁定 PyTorch cu128。
- 完成 CUDA、BF16 与 Blackwell compatibility smoke test。
- 安装 LeRobot training 和最小 PushT runtime。
- 完成 PushT headless 仿真与 ACT 接口验证。
- 定位 OpenCV metadata 和 PushT 图像尺寸两个工程问题。
- 固化 Phase 0 core 配置、检查脚本和实验记录。
- 在 GPU 已占用 4208 MiB 的条件下复核固化脚本，四个 gate 全部通过。

## 思考

- 环境验证要覆盖 tensor contract，而不只是 import 是否成功。
- simulator observation、dataset feature 和 policy feature 必须逐一比对；任何 shape 漂移都会在训练或 rollout 阶段转化为隐蔽错误。
- 第一遍推理延迟主要用于确认代码路径，正式性能必须预热后重复测量并报告分布。
- 51.59M ACT 在本机具备训练可行性，但正式训练显存仍需包含 optimizer、activation、batch 和视频解码后重新测量。
