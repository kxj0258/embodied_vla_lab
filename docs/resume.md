# Resume Material

本文件用于积累秋招项目表述。Phase 0–4 已有冻结实测结果；world model开发结论为NO-GO；Feagine已完成A03运行时与短时动力学诊断，仍不得写成目标到达、策略迁移或真实机器人完成。

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
- 在单张 RTX 5070 Ti Laptop GPU 上完成PushT ACT与Diffusion Policy闭环：固定同一dataset revision和100个held-out seeds，deterministic ACT、VAE ACT、Diffusion EMA分别取得6%、2%、4% success；Diffusion平均max reward从deterministic ACT的0.516提高到0.651，但100-step DDPM推理延迟约增加39倍。
- 搭建 LIBERO 语言条件 MuJoCo 评测闭环；训练 deterministic ACT 100k，并在冻结前选定 checkpoint 后，对未参与开发的 45 个 init states 完整枚举取得 45/45 strict success。
- 在 12 GB RTX 5070 Ti Laptop GPU 上实现 OpenVLA-7B NF4/BF16 推理与跨 Conda 环境哈希 IPC 控制桥；frozen base 在同一 held-out finite set 取得 45/45，最终评测推理中位/P95 约 0.409/0.433 s。
- 冻结OpenVLA base并预注册LIBERO-Goal全10个任务、每任务3个init states的小型多任务闭环集；取得20/30 strict success与10/10任务覆盖，完整保留10个600步timeout并分析复合指令/精细放置薄弱项。
- 按冻结的每任务首成功state完成10/10个原生640渲染复现，逐一匹配原评测步数并制作1920×1080十任务合集；展示样例与20/30可靠性统计分离，所有失败仍保留在结构化结果中。
- 实现 rank-8 QLoRA（27.7M、0.7048% trainable parameters）；paired fullval889 的 loss、token accuracy、continuous L1 分别改善 47.36%、10.33 个百分点、32.63%，但闭环 final 从 base 45/45 回退到 43/45，据此识别 teacher-forced offline objective 与 tail robustness 的错位。
- 实现原生128策略输入与同sim/同EGL直接640渲染的可复现HD展示链路；并排呈现QLoRA典型状态70-vs-221步加速及尾部状态600步timeout-vs-base 68步成功，明确区分控制路径长度、单步推理延迟与整体可靠性。
- 构建冻结配对ego-view实验：直接腕部视角替换在development为0/15；原生wrist rank-8 QLoRA在同一889帧full validation上将loss降低45.77%、token accuracy提高7.73个百分点、continuous L1降低32.43%，但task-7闭环仍为0/3且全部600步timeout，实证揭示teacher-forced改善不等于视角迁移成功。
- 设计 VLA 高层动作到 Feagine PCC IK/controller 的桥接接口，在 MuJoCo 中完成 `[待填任务]` 闭环控制。

## 创新点候选

- 面向刚性机械臂 VLA 到连续体机械臂的 action abstraction 与 controller bridge。
- 将动作 frame、representation、frequency 和 normalization 作为显式数据契约，降低跨环境和跨机器人迁移错误。
- 在受限单 GPU 条件下建立量化推理、参数高效微调和小型世界模型的可复现路线。

## 需要积累的证据

- [ ] 可公开仓库。
- [x] 系统架构图。
- [x] ACT与Diffusion训练指标、checkpoint manifest、100-seed结果和成功rollout video。
- [x] LIBERO单任务冻结held-out finite set success与失败案例。
- [x] OpenVLA latency/VRAM、paired offline与closed-loop对照。
- [x] ACT/OpenVLA真实640渲染视频、Base-vs-QLoRA典型加速/尾部失败并排HD对比与SHA-256 manifest。
- [x] ego-view ablation：camera gate、15-pair直接替换、wrist QLoRA fullval889与3-state闭环NO-GO证据链完整。
- [~] small world model：时序合同及两轮paired 100-update smoke完成；旧全帧模型与新bounded-residual模型均工程PASS但未击败copy-frame，residual conditioned H4还略差于free（1.59303 vs 1.59223）。不得写成有效world model或action-aware dynamics。
- [~] Feagine：从外部目标0/4定位负θ clamp与不可达合同，以672次IK量化1.50 mm回撤；恢复后四方向均达约9.30 mm误差/73.44%进度。进一步定位镜像夹爪step-1硬限位瞬态，并以12条新轨迹证明增大open-stop margin可将±X穿透从约0.2631单调降至0.1150 mm，但仍未过冻结0.1 mm门（严格2/4、无安全命令入选）。可写分层诊断，不得写成VLA迁移或安全恢复完成。
- [x] PushT固定数据/配置、受控run profiles、结构化结果与checkpoint/video SHA-256；Diffusion跨进程轨迹非bitwise确定性的边界已记录。
