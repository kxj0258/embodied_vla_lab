# Embodied VLA Lab

Embodied VLA Lab 是一个面向具身智能算法工程师岗位的长期学习与复现项目。项目以仿真为先，围绕 LeRobot、LIBERO 和 OpenVLA，系统研究机器人模仿学习、视觉语言动作模型、ego-view 操作学习和小型世界模型，并最终迁移到 Feagine A03 连续体机械臂。

> 当前状态：Phase 0——环境兼容性分析与项目初始化。尚未安装依赖、下载数据或运行训练。

## 项目目标

- 理解机器人数据集中的图像、状态、动作和语言表示。
- 跑通 ACT 与 Diffusion Policy 的训练、推理和评测闭环。
- 在 LIBERO 中建立语言条件机器人操作基准。
- 理解 OpenVLA 的视觉编码、语言建模、动作 tokenization 和动作反归一化。
- 在单张 RTX 5070 Ti Laptop GPU 上完成量化推理和小规模参数高效微调实验。
- 研究第三视角到 ego-view 的视角迁移问题。
- 训练小型 action-conditioned latent dynamics 模型。
- 建立 VLA 高层动作到 Feagine PCC IK/控制器的接口。

## 技术路线

```text
PushT + LeRobotDataset
        ↓
ACT image → action chunk
        ↓
Diffusion Policy 对照实验
        ↓
LIBERO language + MuJoCo + demonstrations
        ↓
ACT on LIBERO
        ↓
OpenVLA 4-bit inference / QLoRA feasibility
        ↓
ego-view + viewpoint robustness
        ↓
small latent world model
        ↓
Feagine high-level action → PCC IK/controller → MuJoCo
```

更详细的阶段安排见 [docs/roadmap.md](docs/roadmap.md)，系统边界见 [docs/architecture.md](docs/architecture.md)。

## 三个核心项目的职责

| 项目 | 在本实验室中的职责 |
|---|---|
| LeRobot | 数据格式、策略实现、训练流程、归一化、checkpoint 和统一评测接口 |
| LIBERO | MuJoCo 操作环境、语言任务、专家示范、成功判定和标准 benchmark |
| OpenVLA | VLM backbone、视觉语言条件、动作 tokenization、量化推理和参数高效微调 |

三个上游仓库计划作为固定 commit 的 Git submodule 引入。实验性代码放在 `experiments/`、`configs/` 和 `scripts/`，避免直接大规模修改上游源码。

## 硬件约束

- Ubuntu 22.04.5 LTS
- NVIDIA GeForce RTX 5070 Ti Laptop GPU
- 12 GB VRAM，Blackwell，compute capability 12.0
- 32 GB RAM
- 单 GPU

优先使用 inference、BF16、LoRA/QLoRA、小 batch、梯度累积和小规模仿真实验。禁止从零训练 7B 模型或开展大规模视频模型训练。

## 项目结构

```text
embodied_vla_lab/
├── README.md
├── docs/
│   ├── roadmap.md
│   ├── architecture.md
│   ├── compatibility.md
│   ├── experiment_template.md
│   ├── resume.md
│   ├── phases/
│   └── notes/
├── lerobot/                 # planned Git submodule
├── libero/                  # planned Git submodule
├── openvla/                 # planned Git submodule
├── experiments/
│   ├── act_baseline/
│   ├── diffusion_policy/
│   ├── openvla_finetune/
│   ├── ego_view/
│   ├── world_model/
│   └── feagine_transfer/
├── configs/
├── scripts/
├── checkpoints/
└── results/
```

## 实验记录规范

每个实验必须从 [docs/experiment_template.md](docs/experiment_template.md) 复制一份记录，并至少包含：

- 实验目的与可验证假设
- 代码和环境版本
- 数据集、observation 和 action 定义
- 完整命令与配置
- 训练曲线、成功率、推理速度和显存
- 失败案例与问题定位
- 思考、下一步和可用于简历的材料

W&B 用于在线跟踪，本地同时保存配置、JSON/CSV 指标以及 TensorBoard 日志。模型权重、数据集和大视频不进入 Git，只提交清单、校验信息和发布链接。

## Demo

| Demo | 状态 | 产物 |
|---|---|---|
| PushT ACT | 未开始 | training curve / checkpoint / rollout video |
| ACT on LIBERO | 未开始 | success rate / FPS / failure cases |
| OpenVLA inference | 未开始 | 4-bit latency / VRAM / rollout |
| Ego-view | 未开始 | viewpoint ablation |
| Feagine transfer | 未开始 | instruction-to-controller simulation demo |

所有数值结果将在实验完成后填写，不使用预计结果代替实测结果。

## 开发规则

- 安装依赖前先分析兼容性。
- 不自动运行训练、下载大型模型或启动 GUI。
- 每次修改先说明文件、原因和验证方式。
- 先完成最小 smoke test，再增加数据、任务和模型规模。
- 上游仓库固定 commit；实验代码尽量保持 out-of-tree。
- 真实机器人接入前必须先通过仿真、动作范围和安全约束验证。
