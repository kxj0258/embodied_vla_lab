# Embodied VLA Lab：项目开发说明与 Codex 提示词

## 1. 文档用途

本文档用于交给 Codex，指导其基于现有仓库继续开发 `embodied_vla_lab`，将当前已有的 PushT + ACT + Diffusion Policy 基础，逐步扩展为具备 LIBERO、OpenVLA、统一评测和可开源贡献能力的具身智能项目经历。

仓库地址：<https://github.com/kxj0258/embodied_vla_lab>

## 2. 项目定位

项目名称建议：

> 基于 LeRobot、LIBERO 与 OpenVLA 的具身智能视觉语言动作学习实验平台

项目目标：

- 理解机器人数据集中的图像、状态、语言和动作表示。
- 建立 LeRobot、LIBERO、OpenVLA 之间的数据与动作适配层。
- 完成 ACT、Diffusion Policy 和 OpenVLA 的可复现实验。
- 统一记录成功率、推理延迟、控制频率、显存、checkpoint、seed 和失败案例。
- 研究视角变化、动作表示和模型推理效率对机器人操作的影响。
- 后续建立 VLA 高层动作到 Feagine 连续体机械臂 PCC IK/controller 的桥接。

项目采用 `simulation-first`、`interface-first`、`measurement-first` 原则。实验代码优先放在 `experiments/`、`configs/`、`scripts/` 和 `results/`，避免直接大规模修改上游 submodule。

## 3. 当前已经完成的工作

以下内容已经有仓库文档和结果支撑，可以视为已完成事实：

### 3.1 环境与运行时

- 完成 Python、PyTorch、CUDA、RTX 5070 Ti Laptop GPU 环境适配。
- 完成 LeRobot、PushT、MuJoCo、LIBERO 的基础环境验证。
- 完成 MuJoCo EGL 离屏渲染。
- 完成 LIBERO offline reset/step 和双相机 RGB 渲染验证。
- 将 LeRobot、LIBERO、OpenVLA 作为固定 commit 的 Git submodule 管理。

### 3.2 数据与实验契约

- 固定 PushT dataset revision。
- 完成 206 个 episode 的数据审计。
- 建立 185/21 train/eval split。
- 记录 observation、action、normalization、checkpoint、seed 和视频哈希。
- 建立受控运行 profile 和结构化结果保存方式。
- 设计 canonical observation/action contract，明确图像、状态、语言、动作、坐标系、频率和 normalization 的语义。

### 3.3 ACT 基线

- 完成 VAE ACT 和 deterministic ACT 训练。
- 完成 checkpoint 保存、闭环 rollout 和独立 100-seed 评测。
- deterministic ACT 当前记录：100-seed success 为 6%，平均单 episode 推理时间约 1.17 秒。

### 3.4 Diffusion Policy 基线

- 完成 Diffusion Policy EMA baseline。
- 完成 checkpoint 选择、闭环推理、100-seed 评测和成功视频。
- 当前记录：100-seed success 为 4%。
- Diffusion 的 dense reward 和峰值接近度优于 deterministic ACT，但 100-step DDPM 推理约为 ACT 的 39 倍。
- 当前阶段的有效结论是：连续动作质量指标更好不一定转化为更高的稀疏任务成功率，推理延迟需要单独评估。

## 4. 当前尚未完成的内容

以下内容不能写成已经完成：

- ACT on LIBERO 的完整任务闭环和正式评测。
- OpenVLA 4-bit inference 和 LIBERO rollout。
- OpenVLA 的显存、延迟、动作反归一化和失败分析。
- QLoRA 或其他参数高效微调结果。
- ego-view 视角消融实验。
- world model。
- Feagine 连续体机械臂仿真桥接。
- 已被上游接受的 LeRobot 或 OpenVLA PR。

## 5. 总体开发策略

优先级如下：

1. 先完成候选 A：LeRobot 方向的 LIBERO 评测闭环和统一 benchmark。
2. 再完成候选 B：OpenVLA 的最小语言条件推理和 LIBERO 接入。
3. 在统一评测器基础上做一个受控优化课题。
4. 最后再考虑 OpenVLA 微调、ego-view、Feagine bridge 或上游开源贡献。

不要同时展开多个大型方向。每次只推进一个可验收的 milestone。

## 6. 第一阶段：LIBERO 单任务闭环

### 目标

将当前 PushT 策略复现项目扩展为统一的 LIBERO 机器人操作评测平台。

### 工作内容

1. 固定一个 `libero_spatial` 短任务。
2. 确认并记录：
   - 相机名称、分辨率和裁剪方式；
   - proprioception 定义；
   - 7D action 的坐标系和表达形式；
   - rotation representation；
   - gripper 语义；
   - 控制频率和 action horizon；
   - termination 和 success condition；
   - demonstration revision。
3. 建立 LIBERO demonstration 到统一 contract 的适配器。
4. 将 ACT 接入 LIBERO 单任务 rollout。
5. 实现统一 evaluator，至少记录：
   - success rate；
   - inference latency；
   - control-loop FPS；
   - action horizon；
   - 实际执行动作数量；
   - checkpoint、commit 和 seed；
   - 峰值显存；
   - 失败类别。
6. 保存成功视频、失败视频、JSON/CSV 结果和实验报告。

### 验收标准

```text
固定任务
+ 固定 demonstration
+ 固定 checkpoint
+ 固定 seed
+ 20～50 次 rollout
+ 结构化结果
+ 成功/失败视频
+ 失败原因分类
```

## 7. 第二阶段：OpenVLA 最小推理闭环

### 目标

证明项目可以处理视觉语言动作模型，而不只是调用 ACT/Diffusion Policy。

### 推荐顺序

1. 加载 OpenVLA 预训练模型。
2. 完成单张图片 + 单条语言指令的 4-bit inference smoke test。
3. 阅读并记录 processor、visual encoder、语言输入、action tokenizer 和 `predict_action` 流程。
4. 核对 `unnorm_key`、action dimension、坐标系和控制语义。
5. 将 OpenVLA 输出接入同一个 LIBERO evaluator。
6. 固定一个 LIBERO 任务进行小规模 rollout。
7. 记录显存、单步延迟、控制频率、动作范围和失败样本。

### 重要限制

- 不要一开始追求完整复现官方论文的 LIBERO 结果。
- 不要在 12 GB 显存条件下默认开展大规模 7B 训练。
- 先完成 inference 和数据/动作契约验证，再决定是否做 QLoRA。
- OpenVLA zero-shot 在新任务上可能表现较差，低 success 不应直接视为代码失败。

## 8. 第三阶段：选择一个优化课题

建议只选择一个主线：

### 方向 A：统一动作与评测契约

统一 ACT、Diffusion Policy 和 OpenVLA 的：

- observation 输入；
- action 输出；
- normalization；
- action horizon；
- latency；
- FPS；
- success rate。

这是最稳妥、最适合当前硬件、最容易形成工程成果的方向。

### 方向 B：视角鲁棒性

固定任务、数据量、策略和训练步数，仅改变：

- 第三视角与 ego-view；
- 相机外参；
- 遮挡；
- 图像裁剪；
- 运动模糊；
- action frame。

输出 viewpoint ablation、成功率差异和失败分类。

### 方向 C：VLA 到连续体机械臂桥接

后续实现：

```text
VLA / policy action
        ↓
高层末端目标或 delta pose
        ↓
坐标变换与限幅
        ↓
workspace/collision constraint
        ↓
PCC inverse kinematics
        ↓
连续体机械臂控制器
        ↓
MuJoCo 仿真
```

该方向应在 LIBERO/OpenVLA 闭环稳定后推进，不能直接把刚性机械臂的 joint action 映射到连续体机械臂。

## 9. 推荐的实验记录要求

每个实验都必须保留：

- 实验目的和可验证假设；
- 代码 commit；
- 上游 submodule commit；
- 环境版本；
- 数据集 revision；
- 训练/评测配置；
- 完整运行命令；
- checkpoint 标识；
- seed；
- success rate；
- latency/FPS；
- 显存；
- 成功和失败视频；
- 失败原因；
- 与上一版本的差异；
- 是否可以写入简历；
- 尚未确认的 Claim。

不要把模型权重、数据集和大视频直接提交到 Git。提交配置、manifest、指标摘要、哈希和必要的小型示例。

## 10. 可用于简历的项目经历边界

当前可以写：

> 构建基于 LeRobot、MuJoCo 和 PushT 的具身智能模仿学习实验平台，完成固定数据版本、数据划分、normalization、checkpoint、seed 和视频哈希管理；在单张 RTX 5070 Ti Laptop GPU 上完成 ACT 与 Diffusion Policy 的训练、闭环推理和独立 100-seed 评测，并对比 success、dense reward 和推理延迟，进一步建立 LIBERO/OpenVLA 的语言条件操作评测流程。

在 LIBERO/OpenVLA 完成之前，应使用“正在建立”“正在推进”“计划完成”，不要写“完成了 LIBERO/OpenVLA 闭环”“实现了 VLA 到连续体机械臂控制”等表述。

## 11. 开源贡献策略

优先选择 LeRobot，贡献方向按以下顺序：

1. 文档补充；
2. 复现实验说明；
3. 独立 benchmark/evaluator；
4. 失败轨迹可视化；
5. 环境兼容性修复；
6. 最后再考虑核心算法修改。

贡献前必须确认：

- CONTRIBUTING.md；
- LICENSE；
- 当前 issue 和 PR；
- 是否存在重复实现；
- 最小改动范围；
- 本地复现证据。

不得未经确认直接 fork、push 或提交 PR。

## 12. 给 Codex 的可复制提示词

将下面内容完整复制给 Codex：

```text
你正在协助开发 GitHub 仓库 https://github.com/kxj0258/embodied_vla_lab。

请先阅读仓库 README、docs/roadmap.md、docs/architecture.md、docs/resume.md，以及本开发说明文档。先进行只读代码和文档审计，确认当前真实完成状态，不要把计划、占位或 README 中的未来目标当成已经完成。

项目定位：这是一个基于 LeRobot、LIBERO、OpenVLA 的具身智能视觉语言动作学习实验平台，采用 simulation-first、interface-first、measurement-first 原则。当前已经完成 PushT 数据审计、ACT baseline、Diffusion Policy baseline、固定 seed 评测、checkpoint/video manifest、MuJoCo EGL 和 LIBERO offline runtime 基础验证。当前真实缺口是 LIBERO 单任务闭环、OpenVLA 最小推理闭环、统一 evaluator 和后续可验证优化。

你的工作顺序必须是：

第一步：只读审计
- 检查仓库结构、当前 git 状态、最近提交、已有实验入口、配置、结果和文档。
- 找出已有的 PushT、ACT、Diffusion、LIBERO runtime、OpenVLA 相关代码。
- 明确哪些能力已经完成、哪些只是规划、哪些缺少证据。
- 不运行训练、不下载大模型、不启动 GUI、不修改文件。

第二步：提出一个最小实现计划
- 优先推进候选 A：固定一个 LIBERO-Spatial 短任务�