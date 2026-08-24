# Resume Material

本文件用于积累秋招项目表述。当前所有定量结果均为待填，不得将路线目标写成已经完成的结果。

## 项目名称候选

基于 LeRobot、LIBERO 与 OpenVLA 的具身智能视觉语言动作学习系统

## 项目简介草稿

面向机器人操作与连续体机械臂迁移，构建 simulation-first 的视觉语言动作学习实验平台，统一机器人数据、模仿学习策略、语言条件仿真评测与高层动作控制接口。

## 技术栈

- Python、PyTorch、MuJoCo、Conda
- LeRobot、LIBERO、OpenVLA
- ACT、Diffusion Policy、VLA、LoRA/QLoRA
- W&B、TensorBoard、Git/Git submodule
- PCC kinematics、inverse kinematics、robot control

## 候选项目要点

以下表述必须在结果完成后替换 `[待填]`：

- 设计 LeRobotDataset/LIBERO/OpenVLA 间的数据与动作适配层，统一图像、语言、机器人状态、动作 chunk、坐标系和 normalization 元数据。
- 在单张 RTX 5070 Ti Laptop GPU 上复现 ACT 与 Diffusion Policy，在 `[待填任务]` 上取得 `[待填成功率]`，推理速度 `[待填 FPS]`。
- 搭建 LIBERO 语言条件 MuJoCo 评测流程，完成 `[待填数量]` 个任务、`[待填 episodes]` 次 rollout，并建立 perception/action/temporal failure taxonomy。
- 实现 OpenVLA 4-bit 推理与 `[待填：QLoRA 是否完成]`，将峰值显存控制在 `[待填]`，分析动作 tokenization 与反归一化流程。
- 构建 ego-view 视角对照实验，通过 `[待填方法]` 将 `[待填指标]` 改善 `[待填]`。
- 设计 VLA 高层动作到 Feagine PCC IK/controller 的桥接接口，在 MuJoCo 中完成 `[待填任务]` 闭环控制。

## 创新点候选

- 面向刚性机械臂 VLA 到连续体机械臂的 action abstraction 与 controller bridge。
- 将动作 frame、representation、frequency 和 normalization 作为显式数据契约，降低跨环境和跨机器人迁移错误。
- 在受限单 GPU 条件下建立量化推理、参数高效微调和小型世界模型的可复现路线。

## 需要积累的证据

- [ ] 可公开仓库。
- [ ] 系统架构图。
- [ ] ACT training curve 与 rollout video。
- [ ] LIBERO success rate 与失败案例。
- [ ] OpenVLA latency/VRAM 表。
- [ ] ego-view ablation。
- [ ] Feagine simulation demo。
- [ ] 精确复现环境与命令。
