# Embodied VLA Lab

Embodied VLA Lab 是一个面向具身智能算法工程师岗位的长期学习与复现项目。项目以仿真为先，围绕 LeRobot、LIBERO 和 OpenVLA，系统研究机器人模仿学习、视觉语言动作模型、ego-view 操作学习和小型世界模型，并最终迁移到 Feagine A03 连续体机械臂。

> 当前状态（2026-08-28）：Phase 0--4均已收口，Phase 5A world-model数据门PASS。ACT-on-LIBERO与OpenVLA frozen base在完整45-state held-out final均为45/45 strict success；agentview QLoRA虽改善offline指标但最终仅43/45。Ego-view直接替换为0/15；wrist QLoRA虽改善offline指标，task-7闭环仍为0/3。可靠demo继续使用base；world model尚未训练。

> P0 展示收口已完成：OpenVLA base、ACT成功视频、典型QLoRA加速与init44尾部失败对比、架构图、量化图、SHA-256 manifest、只读核验入口及portfolio/resume材料均已完成。ACT展示capture复用冻结checkpoint/init5并在83步strict success，不改变原45/45 final。

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

完整实测总结见 [docs/project_report.md](docs/project_report.md)，统一证据与复现入口见 [docs/repro_index.md](docs/repro_index.md)。更详细的阶段安排见 [docs/roadmap.md](docs/roadmap.md)，系统边界见 [docs/architecture.md](docs/architecture.md)，当前上游依赖分析见 [docs/dependency_analysis.md](docs/dependency_analysis.md)。

统一 VLA benchmark 契约与历史结果导入入口见 [docs/benchmark_contract.md](docs/benchmark_contract.md)。只读核验示例：

```bash
python3 scripts/benchmark.py import results/phase2b_act_libero_ckpt100000_final_heldout45.yaml
python3 scripts/benchmark.py import results/phase3a_openvla_libero_final_heldout45_v1.yaml
python3 scripts/benchmark.py compare --result \
  results/phase2b_act_libero_ckpt100000_final_heldout45.yaml \
  results/phase3a_openvla_libero_final_heldout45_v1.yaml
```

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
| LIBERO Phase 2A task | 静态/demo/40:10数据合同PASS；task replay成功，精确sim-state replay FAIL（已接受边界） |
| LIBERO ACT GPU gate | 34.19M参数；两步loss 0.6174→0.4564；峰值allocated VRAM 673.6 MiB，PASS |
| LIBERO ACT overfit 200 | loss 0.6174→0.1750（-71.65%），未达到预注册>75%下降，FAIL |
| LIBERO ACT overfit 400 | loss 0.6174→0.1356（-78.04%），最小0.1229，PASS |
| LIBERO ACT fullsplit smoke v1 | 20步优化完成；checkpoint序列化兼容错误，FAIL，保留输出 |
| LIBERO ACT deterministic 100k | PASS；100k steps；validation 101次完整覆盖889 frames；最低0.42663@1k、final 0.44003；rollout未运行 |
| LIBERO ACT step-100k dev rollout | PASS；固定init states 0–4为5/5 strict success；约3.49 s/episode；development only |
| LIBERO ACT step-4,460 dev rollout | 评测PASS；同5 states为4/5，state 4在600步超时；低于step-100k开发结果 |
| LIBERO ACT checkpoint selection | step 49,060与100k均5/5、reward 1.0；按403<407总步数冻结step-100k；选择未被final回改 |
| LIBERO ACT held-out final | step-100k在全部未用于开发的init states 5–49上45/45 strict success；mean reward 1.0；Phase 2B完成 |
| OpenVLA environment | 独立Python 3.10环境；cu128、BF16、NF4 `sm_120` kernel及本地推理类收窄导入PASS |
| OpenVLA checkpoint | 官方LIBERO-Goal revision `fa5ae1e...9baf`；16文件、15,085,050,797逻辑字节逐文件哈希PASS |
| OpenVLA 4-bit single image | 冻结真实LIBERO图像7D动作PASS；load 5.806 s、单次生成1.024 s、峰值allocated/reserved VRAM 4.71/4.84 GiB |
| OpenVLA isolated step bridge | PASS；固定init state 0完成1 reset/1 prediction/1 env step；reward 0、未终止，峰值VRAM约4.71 GiB |

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
| ACT on LIBERO | Phase 2B完成；冻结step-100k；45-state held-out final为45/45 strict success | 配置、训练/验证曲线、checkpoint manifest、held-out final、83-step headless success demo |
| OpenVLA inference | Phase 3A完成；development 5/5，held-out states 5–49为45/45 strict success | median/P95推理0.409/0.433 s；state 12长尾593步 |
| OpenVLA QLoRA | Phase 3B完成：offline改善、dev5 5/5，但heldout仅43/45（base 45/45） | 工程可行但鲁棒性回退；adapter仅研究资产，可靠demo继续使用冻结base |
| Ego-view | Phase 4完成；camera pipeline PASS，direct substitution与wrist-adapter recovery均NO-GO | 15-pair视角消融、889帧offline对照、3-state闭环失败、证据链图 |
| Small world model | Phase 5A/5B PASS；Phase 5C旧模型NO-GO；Phase 5D residual静态PASS；Phase 5E执行PASS但action conditioning开发NO-GO | residual H4 MAE：copy 1.59125、free 1.59223、conditioned 1.59303；不进入确认集或正式训练 |
| Feagine transfer | Phase 6C zero-step PASS；Phase 6D动态v1保留6/7 FAIL，逐关节v2诊断PASS | 唯一越界为夹爪slide软限位穿透11.77 μm，低于预注册0.1 mm材料阈值；禁止据此宣称target-reaching、VLA或硬件成功 |

Phase 6E首个A03 bounded move-to-target development smoke执行9/9 PASS：53.90 mm目标经400步/2秒后误差9.50 mm，下降82.37%；IK 4次迭代，材料越界为零。该成功仅覆盖同一PCC core生成的单目标，不含VLA、相机、任意工作区目标、reward benchmark或硬件。

Phase 6F外部显式±X/±Y 35 mm目标执行0/4 FAIL：±X在IK阶段失败且末端不动；±Y虽IK数值成功，MJCF最终误差却扩大到241.63/271.86 mm。四条状态均有限、时间准确、无材料越界和接触，定位为target→section/controller映射失败而非动力学崩溃。禁止换目标、放宽阈值或覆盖重跑。

Phase 6G同目标映射诊断完成：±X从全零直杆初值陷入IK奇异点并输出全零controls；±Y的core解包含负θ，preset直接clamp为0后破坏原解。clamp后core预测与MJCF实际位移cosine仍为0.9994/0.9993，说明actuator/dynamics忠实跟随错误的applied配置。Phase 6F仍为0/4；修复候选是signed-θ等价规范化与冻结multi-start IK，尚未执行。

Phase 6H CPU数学门4/5 FAIL：signed-θ规范化在260组FK上以≤9.22e-8 m/4.28e-7误差证明等价，但8个multi-start seeds对原四目标的最佳残差仍稳定在1.592–1.615 mm，未达到0.1 mm IK门槛。原目标在近最大轴向伸展处增加35 mm横移且保持方向，疑似几何不可达；不得放宽容差或把近似解当成功。

Phase 6I冻结0–5 mm/0.25 mm轴向回撤扫描，672次CPU IK显示±X/±Y均在1.50 mm首次进入原0.1 mm容差，方向spread=0。v1因浮点列表精确比较记5/6 FAIL并保留；不重算IK的v2包装恢复10/10 PASS。该结果只定义后续可达目标合同，尚未运行MJCF恢复。

Phase 6J MJCF recovery为2/4 FAIL：四方向IK、末端误差约9.30 mm和进度约73.44%均通过，±Y完整PASS；±X仅因slide穿透0.2631 mm超过冻结0.1 mm材料阈值失败。hinge越界和contact均为0。说明目标/表示恢复有效，但X平面安全门未过；禁止放宽阈值或覆盖重跑。

Phase 6K诊断完成：+X的`feagine_grasper_joint_0`与−X镜像的`joint_1`均在step 1从完全张开下限−13 mm瞬态到约−13.2631 mm；另一指仅约2.45 μm，±Y仅10⁻¹¹ m量级。根因是夹爪目标正置硬下限时的方向相关首步惯性瞬态，不是PCC proxy或持续发散。Phase 6J仍为2/4。

Phase 6L执行PASS但选型NO-GO：预注册0.00/0.01/0.02/0.05均为2/4。±X slide穿透随command单调由约0.2631降至0.1150 mm，±Y均PASS，约9.30 mm末端误差和73.44%进度不变；0.05仍超过冻结0.1 mm阈值约0.015 mm，因此未选择任何安全margin。不得后验新增数值或回改Phase 6J 2/4。

所有数值结果将在实验完成后填写，不使用预计结果代替实测结果。

2026-09-04完成经授权的推荐存储精简：删除550个可再生训练恢复/非关键周期payload，共26.080 GiB，项目
由约36 GiB降至9.4 GiB。所有结果、raw log、展示资产、失败目录及8个关键推理模型/adapter均保留；完整
删除前路径、大小和SHA-256见`results/storage_prune_recommended_20260904_manifest.yaml`。被删除的optimizer
state和周期checkpoint不能再从当前本地目录直接恢复训练，旧manifest仅保留其删除前身份。

P0 canonical presentation manifest：`results/p0_mvp_v2_manifest.yaml`。本地媒体位于
`outputs/presentation/p0_mvp_v2_20260827/`，可用
`/home/kxj/miniforge3/envs/evla-core/bin/python scripts/verify_p0_assets.py` 离线核对源证据、视频/图片哈希和
H.264 媒体结构。展示视频是冻结观测帧的离线重放，10 fps 仅为展示帧率，不代表控制频率或模型吞吐。

P0-HD 已COMPLETE：ACT与OpenVLA base init5成功视频、init40 base/QLoRA 221/70步典型加速、init44 base 68步成功对QLoRA 600步timeout均使用真实640×640模拟器渲染。已生成两个1920×1080并排研究对比视频，位于`outputs/presentation/p0_hd_comparisons_v1_20260828/`，结构化manifest为`results/p0_hd_comparisons_v1_manifest.yaml`。QLoRA单步推理更慢且总体held-out仍为43/45低于base 45/45；不允许基于尾部失败调参或重跑final。

P0-Breadth的冻结OpenVLA base评测覆盖LIBERO-Goal全10个任务目录，每任务3个固定init states：总体20/30 strict success，且10/10任务至少成功1次。失败为10个600步timeout，已全部保留；复合抽屉、奶油奶酪入碗和碗放盘仅为1/3。这是固定30-episode小型展示集，不是官方完整LIBERO benchmark；不允许删除失败、替换init state或选择性重跑。

P0-Breadth现已COMPLETE：按预先冻结的“每任务首个成功state”规则完成10/10个真实640渲染recapture，步数逐一复现且合计1,296步；已离线生成1920×1080、20 fps、64.8秒的十任务合集`outputs/presentation/p0_breadth_10task_montage_v1_20260828/openvla_10task_success_montage_hd.mp4`。Task 6原presentation编码损坏已保留为非规范资产，并从完整clean master离线重组为规范版本。展示成功样例不改变冻结总体20/30结论。

Phase 4A viewpoint camera gate现为`RUNTIME_PASS`：冻结OpenVLA base不变，后续仅配对替换单图输入源`agentview`/`eye_in_hand`。开发任务`[0,3,6,7,8]`与最终任务`[1,2,4,5,9]`在任何eye-in-hand结果出现前冻结，每任务3个init、每视角各一次，形成30回合development与30回合final。

Phase 4A运行时v1因将观测别名`eye_in_hand`误作MuJoCo camera名而FAIL；实际名称为`robot0_eye_in_hand`。该次为0 step/0 prediction且清理正常，失败目录/log完整保留。仅修正相机名、使用全新路径的v2静态9项PASS，checkpoint、任务split和研究边界均未改变。

Phase 4A v2运行时现已PASS：一次reset、0 step、0模型预测；agentview与eye-in-hand在原生128及直渲染640下均为有效非恒定RGB，跨视角MAE分别76.46/77.50，四张图像hash与视觉QA通过，环境正常关闭。这只证明双视角输入链路可行，不代表OpenVLA在ego-view下成功。

Phase 4B development paired rollout已完成：任务`[0,3,6,7,8]`×init `[0,1,2]`×视角`[agentview,eye_in_hand]`共30回合，形成15个同task/init/seed配对。使用实际冻结revision `fa5ae1e750...9baf`、单次模型加载和每episode显式init环境；禁止训练、checkpoint选择、失败删除和选择性重跑。

Phase 4B执行PASS但直接ego-view替换为development NO-GO：agentview为10/15 strict success并覆盖5/5任务，eye-in-hand为0/15且0/5任务覆盖，15个wrist回合全部600步timeout。配对结局为10 agent-only、0 ego-only、5 neither，双侧精确配对p=0.001953。该结果揭示冻结第三视角VLA不能零适配迁移到腕部视角；final tasks `[1,2,4,5,9]`仍未触碰，也不会为失败假设直接运行。

Phase 4B规范量化图位于`outputs/presentation/phase4b_viewpoint_dev_summary_v2_20260828/`，manifest为`results/phase4b_viewpoint_dev_summary_v2_manifest.yaml`；可用`scripts/verify_phase4b_assets.py`只读核验result、raw log、13,663行trajectory及图像hash。

Phase 4C已冻结development-only wrist适配假设：复用turn-on-stove 50 demos及40/10 split，仅把训练图像源换为原生`eye_in_hand_rgb`。数据门确认3,571 train/889 validation帧与有限7D action；固定20-train/10-val腕部fixture已hash，rank-8 QLoRA 20-update smoke静态PASS待宿主。该smoke只验证优化/离线响应，不能证明闭环恢复或触碰final tasks。

Phase 4C 20-update wrist QLoRA smoke执行PASS，adapter/optimizer/RNG完整；val10 loss从13.704降至9.279（-32.29%），但action-token accuracy从16/70降至13/70（-4.29个百分点）。因此仅判定工程可行、offline信号混合，不选择checkpoint、不运行闭环；下一门须先做冻结889帧完整wrist validation。

Phase 4D wrist fullval889现为`PAIRED_OFFLINE_COMPLETE_ENVIRONMENT_GATE_PENDING`：在同一889帧/6,223-token fixture上，step-20 wrist QLoRA相对base将loss降低45.77%、token accuracy提高7.73个百分点、continuous L1降低32.43%，且10/10 validation episodes三项均改善。代价是吞吐下降13.28%、中位/P95延迟约增加15.7%/15.4%、allocated/reserved VRAM增加5.82%/4.68%。结论仅为wrist offline adaptation PASS；环境恢复、success和reward仍未知，final tasks仍未触碰。

Phase 4D规范配对结果为`results/phase4d_openvla_wrist_step20_vs_base_fullval889_paired.yaml`，量化图manifest为`results/phase4d_wrist_fullval_summary_v1_manifest.yaml`。step-20只可进入新冻结的development闭环门，不是可靠策略选择，也不得据此运行final。

Phase 4E development-only wrist闭环门静态12/12 PASS：只评测训练任务7的step-20 adapter，复用Phase 4B已冻结的同init `[0,1,2]`/seed base 0/3证据，不重复运行base。预注册adapter 0/3为NO-GO、1–2/3为partial recovery、3/3为strong recovery；任何结果均不自动授权final tasks或跨任务泛化结论。

Phase 4E执行PASS但闭环结论为`NO_GO_NO_CLOSED_LOOP_RECOVERY`：step-20 adapter在task 7 wrist init `[0,1,2]`仍为0/3，三个回合均600步timeout、reward 0，与base形成3个neither-success配对。1,800次动作均有限且在范围内，worker完整关闭；adapter平均推理0.4876 s，比同配对base约慢43.0%。Phase 4D的强offline改善未转化为闭环恢复；禁止据此调参、换checkpoint、选择性重跑或触碰final。

Phase 4E的offline-vs-closed-loop规范图由`results/phase4e_wrist_no_go_summary_v1_manifest.yaml`索引；它把离线改善与0/3环境结果并列展示，避免在项目材料中混淆两类证据。

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
