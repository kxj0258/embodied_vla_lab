# Embodied VLA Lab Roadmap

## 总体原则

项目采用 simulation first、interface first、measurement first 的路线。每个阶段都有明确 gate；上一阶段的数据契约、动作语义或评测闭环未通过时，不增加模型规模。

目标时间为 2026 年 10–11 月，每天可投入约 3–4 小时。路线分为秋招 MVP 和后续增强两部分。

## 秋招 MVP：8 周

| 周次 | 阶段 | 核心任务 | Gate / 可交付产物 |
|---|---|---|---|
| Week 1 | Phase 0 | 驱动、PyTorch、MuJoCo、EGL、项目环境 | CUDA tensor、离屏渲染、环境锁定文档 |
| Week 2 | Phase 1A | 读取 PushT 数据；分析 observation、action、时间窗口和 normalization | 单样本检查报告、ACT 配置说明 |
| Week 3 | Phase 1B | 单 GPU ACT 训练与 rollout | 曲线、checkpoint、视频、成功指标 |
| Week 4 | Phase 1C/2A | Diffusion Policy 对照；LIBERO 单任务接入 | 同数据对照、LIBERO reset/step/video |
| Week 5 | Phase 2B | ACT on one LIBERO-Spatial task | 20–50 rollouts、成功率、FPS、失败分类 |
| Week 6 | Phase 3A | OpenVLA 4-bit pretrained inference | 峰值显存、单步延迟、动作检查、rollout |
| Week 7 | Phase 3B/4 | QLoRA 可行性探测或 ego-view 视角消融 | go/no-go 结论、视角对照结果 |
| Week 8 | MVP 收口 | Demo、README、架构图、简历表述和复现实验检查 | 可展示仓库与一键复现说明 |

## Phase 0：环境准备

目标：建立两个相互隔离的环境，并首先完成 `evla-core`。

- `evla-core`：LeRobot、PushT、ACT、Diffusion Policy、LIBERO、MuJoCo。
- `evla-openvla`：OpenVLA 量化推理和 QLoRA；到 Phase 3 再创建。

Gate：

- PyTorch 识别 RTX 5070 Ti 和 compute capability 12.0。
- CUDA tensor 运算正常。
- MuJoCo EGL 离屏渲染得到有效 RGB 图像。
- LeRobot 和 LIBERO 最小 import/reset/step 正常。
- 精确记录 Python、PyTorch、CUDA runtime、驱动和 commit。

## Phase 1：LeRobot 与 ACT

先使用 PushT，避免同时引入语言、MuJoCo、复杂相机和长时序控制。

学习重点：

- LeRobotDataset v3 目录与 episode 索引。
- 图像/state/action feature key。
- action chunk 与 padding mask。
- normalization、preprocessor 和 postprocessor。
- ACT encoder/decoder、latent variable 和 temporal aggregation。

输出：训练曲线、checkpoint、rollout video、成功指标、峰值显存和失败案例。

## Phase 2：LIBERO Simulation

从 `libero_spatial` 的一个短任务开始，再扩展任务数量。

必须确认：

- 相机图像名称、分辨率和裁剪。
- 机器人 proprioception。
- 7D action 的坐标系、旋转表示、gripper 语义和控制频率。
- episode termination 与 success condition。
- demonstration 与 evaluation 环境的一致性。

输出：成功率、推理 FPS、控制频率、失败分类和视频。

## Phase 3：OpenVLA

顺序：

1. 阅读 processor、visual encoder、LLM、action tokenizer 和 `predict_action`。
2. 只下载一个与目标 suite 对应的 checkpoint。
3. 4-bit 单图推理。
4. 4-bit LIBERO 闭环评测。
5. batch size 1 的 QLoRA 显存探测。
6. 通过 go/no-go 后才进行小数据微调。

普通 LoRA 的显存要求超过本机预算，因此不把它设为本机必达指标。QLoRA 也必须经过峰值显存和数值稳定性验证。

## Phase 4：Ego-view Extension

研究变量：相机外参、遮挡、尺度、运动模糊、图像增强、动作坐标和 normalization。

优先做受控对照：固定任务、数据量、策略和训练步数，只改变 camera setup。

## Phase 5：Small World Model

不训练大规模视频生成模型。优先选择：

```text
image_t → frozen/small encoder → latent_t
latent_t + action_t → latent_{t+1:t+H}
```

比较复制当前帧、无动作模型和 action-conditioned 模型，评估单步与多步误差，以及是否能支持简单候选动作排序。

## Phase 6：Feagine Transfer

建议接口：

```text
image + instruction
        ↓
VLA high-level action / target
        ↓
Feagine action adapter
        ↓
tip target + constraints
        ↓
PCC IK/controller
        ↓
MuJoCo continuum robot
```

第一版任务：Move to target。随后增加 obstacle avoidance 和 contact interaction。

## 11 月增强目标

- 完成小型 world model。
- 建立 Feagine simulation bridge。
- 完成至少一个语言指令到连续体控制闭环 Demo。
- 将关键实验整理成项目报告、演示视频和简历材料。
