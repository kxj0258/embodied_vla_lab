# Embodied VLA Lab

Embodied VLA Lab 是一个面向具身智能算法工程师岗位的长期学习与复现项目。项目以仿真为先，围绕 LeRobot、LIBERO 和 OpenVLA，系统研究机器人模仿学习、视觉语言动作模型、ego-view 操作学习和小型世界模型，并最终迁移到 Feagine A03 连续体机械臂。

> 当前状态（2026-08-26）：Phase 0、Phase 1A、Phase 1B ACT和Phase 1C Diffusion Policy PushT baseline均已完成并本地提交。固定206-episode PushT snapshot已完成185/21 split；ACT与Diffusion均完成训练、checkpoint、固定seed闭环、独立100-seed评测和成功视频。MuJoCo EGL与LIBERO offline reset/step/双相机渲染已验证；下一阶段是固定一个LIBERO任务与demonstration合同并建立训练评测闭环。

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

更详细的阶段安排见 [docs/roadmap.md](docs/roadmap.md)，系统边界见 [docs/architecture.md](docs/architecture.md)，当前上游依赖分析见 [docs/dependency_analysis.md](docs/dependency_analysis.md)。

## 三个核心项目的职责

| 项目 | 在本实验室中的职责 |
|---|---|
| LeRobot | 数据格式、策略实现、训练流程、归一化、checkpoint 和统一评测接口 |
| LIBERO | MuJoCo 操作环境、语言任务、专家示范、成功判定和标准 benchmark |
| OpenVLA | VLM backbone、视觉语言条件、动作 tokenization、量化推理和参数高效微调 |

三个上游仓库已作为浅克隆 Git submodule 固定到具体 commit。实验性代码放在 `experiments/`、`configs/` 和 `scripts/`，避免直接大规模修改上游源码。

## 硬件约束

- Ubuntu 22.04.5 LTS
- NVIDIA GeForce RTX 5070 Ti Laptop GPU
- 12 GB VRAM，Blackwell，compute capability 12.0
- 32 GB RAM
- 单 GPU

优先使用 inference、BF16、LoRA/QLoRA、小 batch、梯度累积和小规模仿真实验。禁止从零训练 7B 模型或开展大规模视频模型训练。

## 当前验证环境

| Component | Version / result |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| LeRobot | 0.6.2, editable pinned submodule |
| PushT | 0.1.6, headless reset/step/render PASS |
| GPU architecture | compute capability 12.0 / `sm_120` |
| ACT interface | image/state → `[1,100,2]` action chunk PASS |
| PushT dataset | revision `7628202a...`, 206 episodes, 185/21 split |
| Deterministic ACT 30k | 6/100 success，1.17 s/episode |
| VAE ACT 25k | 2/100 success |
| Diffusion EMA 30k | 4/100 success，45.55 s/episode |
| MuJoCo / LIBERO | EGL physics、offline reset/step、双相机RGB PASS |

环境决策、安装顺序和已知例外见 [Phase 0 记录](docs/phases/phase_00_environment.md) 与 [兼容性矩阵](docs/compatibility.md)。宿主侧复核命令：

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab
conda activate evla-core
python scripts/check_phase0_core.py
```

该检查默认离线、无 GUI、不训练；PushT 96×96 contract 固定在 [configs/env/pusht_96.yaml](configs/env/pusht_96.yaml)。

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
├── lerobot/                 # pinned Git submodule
├── libero/                  # pinned Git submodule
├── openvla/                 # pinned Git submodule
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
| PushT deterministic ACT | 完成；独立100-seed为6% success | 配置、checkpoint manifest、结构化结果、成功视频 |
| PushT Diffusion Policy | 完成；独立100-seed为4% success，dense reward优于ACT但推理约慢39倍 | EMA checkpoint、配对统计、两个成功视频 |
| ACT on LIBERO | 未开始 | success rate / FPS / failure cases |
| OpenVLA inference | 未开始 | 4-bit latency / VRAM / rollout |
| Ego-view | 未开始 | viewpoint ablation |
| Feagine transfer | 未开始 | instruction-to-controller simulation demo |

所有数值结果将在实验完成后填写，不使用预计结果代替实测结果。

## PushT复现入口

关键训练与评测不再只依赖对话或被忽略的raw log。受控profile入口为：

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab
conda activate evla-core
scripts/runs/pusht.sh --list
EVLA_DRY_RUN=1 scripts/runs/pusht.sh diffusion-30k-train
```

脚本默认严格离线，并拒绝覆盖已有log或output；选择训练profile仍不等于获得训练授权。完整的命令资产审计、
profile到结构化结果的映射、生成文件边界和随机Diffusion轨迹的复现限制见
[PushT reproducibility audit](docs/pusht_reproducibility.md)。ACT细节见
[ACT baseline](experiments/act_baseline/README.md)，Diffusion细节见
[Diffusion Policy baseline](experiments/diffusion_policy/README.md)。

## 开发规则

- 安装依赖前先分析兼容性。
- 不自动运行训练、下载大型模型或启动 GUI。
- 每次修改先说明文件、原因和验证方式。
- 先完成最小 smoke test，再增加数据、任务和模型规模。
- 上游仓库固定 commit；实验代码尽量保持 out-of-tree。
- 真实机器人接入前必须先通过仿真、动作范围和安全约束验证。
