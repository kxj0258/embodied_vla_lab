# Embodied VLA Lab — Phase 0 Core Handoff

更新时间：2026-08-25  
用途：供新的 Codex/ChatGPT 对话快速恢复上下文并继续项目。  
项目根目录：`/home/kxj/embodied_ws/embodied_vla_lab`

## 1. 项目目标

Embodied VLA Lab 是一个面向具身智能算法工程师秋招的长期学习与复现项目，围绕：

1. ego-view robot imitation learning；
2. Vision-Language-Action；
3. simulation-based robot learning；
4. embodied world model；
5. 最终迁移到 Feagine A03 连续体机械臂。

路线坚持 simulation first、interface first、measurement first。最终控制链路设想：

```text
image + instruction
  → VLA high-level action / target
  → Feagine action adapter
  → PCC IK / controller
  → MuJoCo continuum robot
```

用户背景：自动化/机器人、MuJoCo、连续体机器人、RL、机器人控制和 PCC 运动学；已有 Feagine continuum robot simulation 项目。

## 2. 用户与协作约束

- 中文交流。
- 用户每天约投入 3–4 小时，目标在 2026 年 10–11 月形成完整项目。
- 当前协作方式：助手先解释原因、源码结构和关键模块，再给出命令；用户在宿主终端执行并粘贴原始输出，助手根据结果分析后再推进。
- 安装、下载、GPU、训练和 GUI 命令由用户执行。
- 不自动运行训练。
- 不自动下载大型数据、模型或几十 GB checkpoint。
- 不自动启动 GUI 仿真。
- 不自动安装大量依赖。
- 修改文件前说明：修改文件、修改原因、如何验证。
- 环境/文档变更与 Git commit 分开授权。
- 每个阶段更新 docs，至少记录目的、环境、命令、结果、问题和思考。
- 不使用多 GPU，不从零训练 7B，不训练大规模视频模型。
- 优先 inference、BF16、LoRA/QLoRA、小 batch、梯度累积、小规模仿真。

账号信息：

- GitHub：`kxj0258`，当前项目尚无 GitHub remote。
- Hugging Face：`kxj25`，用户允许后续上传。
- Git identity：`kxj_jiaolong <kxj1225@sjtu.edu.cn>`。
- 实验追踪计划：W&B + 本地 TensorBoard/结构化日志。

## 3. 硬件与宿主环境

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-138-generic |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| VRAM | 12227 MiB |
| GPU architecture | Blackwell, compute capability 12.0 / `sm_120` |
| Driver | 580.173.02 |
| Driver CUDA ceiling | 13.0 |
| RAM | 32 GB（用户提供） |
| Conda | Miniforge / conda 26.5.3 |
| Desktop | X11 |
| Initial free disk | 约 126 GB |

重要解释：

- `nvidia-smi` 的 CUDA 13.0 是驱动上限，不表示安装了 CUDA 13 Toolkit。
- 当前 PyTorch wheel 自带 CUDA 12.8 runtime。
- 系统没有 `nvcc`，当前不阻塞；只有必须编译 CUDA extension 时再评估 Toolkit。
- Codex 沙箱看不到 `/dev/nvidia*`；GPU 证据必须由用户宿主终端返回。
- Unitree 等程序可能占用 GPU。不要默认要求永久关闭，GPU 操作前先用 `nvidia-smi` 检查余量。

## 4. 仓库与 Git 状态

项目仓库：

```text
/home/kxj/embodied_ws/embodied_vla_lab
```

当前分支：`main`。

截至创建本 handoff 前，已提交历史：

```text
e88d63f chore: lock phase 0 core environment
dc008d3 build: add pinned upstream submodules
5d5d04e docs: initialize embodied VLA lab
```

三个浅克隆 submodule：

| Repository | Path | Commit |
|---|---|---|
| LeRobot | `lerobot/` | `8b256a6c0d4769c3cc3e7e98f04940126398a391` |
| LIBERO | `libero/` | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| OpenVLA | `openvla/` | `c8f03f48af692657d3060c19588038c7220e9af9` |

本 handoff 文件是 commit `e88d63f` 之后新增的，尚未提交。新对话开始时先运行：

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab
git status --short --branch
git log --oneline --decorate -5
git submodule status
```

不要假设工作区仍然干净，也不要覆盖用户后续产生的文件。

## 5. 三个上游仓库的职责与依赖决策

### LeRobot

作为工程主干：dataset format、observation/action features、processor、ACT、Diffusion Policy、training、checkpoint 和 evaluation。

Pinned LeRobot 版本为 0.6.2，要求 Python `>=3.12`、PyTorch `>=2.7,<2.12`，Linux PyTorch 源面向 cu128。

### LIBERO

用于 MuJoCo language-conditioned manipulation benchmark、demonstration、task suite、success condition 和 evaluation protocol。

原始 LIBERO requirements 面向 Python 3.8、PyTorch 1.11、CUDA 11.3、旧 NumPy，不能直接混入当前 Blackwell 环境。原始 submodule 用于理解 benchmark 源码和语义；实际 runtime 计划优先使用 LeRobot 当前 `libero` extra 的 `hf-libero` compatibility package。

### OpenVLA

用于理解 VLM backbone、visual encoder、action tokenization、`predict_action`、动作反归一化和 LoRA/QLoRA。

Pinned OpenVLA 原始依赖固定 PyTorch 2.2、torchvision 0.17、Transformers 4.40.1、timm 0.9.10、tokenizers 0.19.1 和旧 FlashAttention。不能与 LeRobot 当前环境混装。

Phase 3 单独创建 `evla-openvla`：

- Python 3.10；
- Blackwell-compatible cu128 PyTorch；
- 4-bit inference first；
- batch size 1；
- eager/SDPA first；
- 第一轮不安装 FlashAttention；
- 不直接普通执行 `pip install -e .`；
- 标准 LoRA 显存不适合本机，QLoRA 先做 go/no-go。

## 6. evla-core 已安装环境

Conda environment：

```text
/home/kxj/miniforge3/envs/evla-core
```

关键版本：

| Package | Version |
|---|---|
| Python | 3.12.14 |
| pip | 26.2.1 |
| FFmpeg | 7.1.1 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| LeRobot | 0.6.2 editable，指向 pinned submodule |
| gymnasium | 1.3.0 |
| gym-pusht | 0.1.6 |
| opencv-python-headless | 4.13.0.92 |
| datasets | 4.8.5 |
| av | 15.1.0 |
| torchcodec | 0.11.1 |
| accelerate | 1.14.0 |
| wandb | 0.27.2 |

环境大小在安装 LeRobot training 依赖后约 8.2 GB。

关键安装顺序：

```bash
conda activate evla-core

python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

cd /home/kxj/embodied_ws/embodied_vla_lab/lerobot
python -m pip install -e ".[training]"

python -m pip install \
  pygame==2.6.1 \
  pymunk==6.11.1 \
  shapely==2.1.2 \
  scikit-image==0.26.0

python -m pip install --no-deps gym-pusht==0.1.6
```

环境基础层文件：`configs/env/evla-core.yml`。该 YAML 只描述 conda 基础层，不是假装完整的一键安装文件。

## 7. 已完成验证与实测结果

### CUDA / BF16

- `torch.cuda.is_available() == True`。
- GPU 名称正确。
- compute capability：`(12,0)`。
- PyTorch architecture list 包含 `sm_120`。
- BF16 supported。
- 4096×4096 BF16 matmul finite：PASS。
- 一次实测 matmul：0.103391 s，峰值 PyTorch allocation 184.12 MiB，仅作为 smoke test。

早期空闲桌面基线约使用 1 GB VRAM、空闲约 10.7 GB。

### PushT headless simulator

无 GUI reset、10 个随机 step、`rgb_array` render：PASS。

实际 contract：

| Item | Value |
|---|---|
| raw image | `uint8[96,96,3]` |
| raw state | `float64[2]` agent position |
| action | `float32[2]` absolute target position |
| action range | 每维 `[0,512]` |
| control rate | 10 Hz |
| visualization frame | `uint8[384,384,3]` |

### LeRobot preprocessing

```text
pixels uint8[B,H,W,C]
  → observation.image float32[B,C,H,W], range [0,1]

agent_pos float64[B,2]
  → observation.state float32[B,2]
```

### ACT interface

随机初始化、无预训练权重、无数据、无训练：

| Item | Result |
|---|---|
| image input | `float32[1,3,96,96]` |
| state input | `float32[1,2]` |
| action chunk output | `float32[1,100,2]` |
| `chunk_size` | 100 |
| `n_action_steps` | 10 |
| backbone | ResNet18 |
| parameters | 51.59 M |
| manual first-pass | 0.309598 s / 222.56 MiB |
| repeat script first-pass | 0.282970 s / 223.43 MiB |

两个延迟都包含首次 warm-up，不是正式 benchmark；随机动作没有控制意义，未送入 simulator。

### 固化脚本

文件：`scripts/check_phase0_core.py`。

用户在宿主终端执行：

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab
conda activate evla-core
python scripts/check_phase0_core.py
```

结果：

```text
[1/4] Package contract: PASS
[2/4] CUDA and BF16: PASS
[3/4] LeRobot PushT wrapper: PASS
[4/4] ACT image-to-action interface: PASS
Phase 0 core smoke test: PASS
```

复核前 GPU 已使用 4208 MiB、空闲 7540 MiB、利用率 58%，因此重复延迟仍不能作为性能对照。

## 8. 两个必须保留的已知问题

### OpenCV metadata 例外

LeRobot core 需要 `opencv-python-headless`，而 `gym-pusht==0.1.6` metadata 要求 `opencv-python`。两者提供同一个 `cv2` namespace，不应共存。

当前有意保留 headless wheel，并使用 `gym-pusht --no-deps`。因此：

```bash
python -m pip check
```

预期有且仅有：

```text
gym-pusht 0.1.6 requires opencv-python, which is not installed.
```

不要为了消除此提示安装 `opencv-python`。如果出现其他 dependency error，必须单独分析。

### PushT 384/96 feature mismatch

Pinned LeRobot `PushtEnv` 默认声明 image feature 为 `(3,384,384)`，但其 `gym_kwargs` 没有向 `gym-pusht==0.1.6` 传 observation width/height，实际 runtime 为 `(3,96,96)`。

项目不修改上游 submodule，而是在：

```text
configs/env/pusht_96.yaml
```

显式声明 96×96，并通过固化脚本验证 declared/runtime 一致。

## 9. 其他排查记录

- 第一次 ACT 检查在模型创建前调用 `torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))`，出现 `RuntimeError: Invalid device argument`。
- 修复方式：先 `torch.cuda.set_device(0)`，创建一个极小 CUDA tensor 初始化上下文，再给 CUDA memory API 传整数设备编号 `0`。
- 宿主没有 `rg`，不影响项目；命令检查时可用纯 Python 或 `grep` 代替，不需要为此立即安装 ripgrep。
- Codex 沙箱运行 `conda list` 时曾尝试清理 pip 覆盖后的旧 conda metadata，但沙箱阻止写入，环境未改变。如果宿主也出现类似警告，先回传原始输出，不要手动删除 conda-meta 文件。

## 10. 重要项目文件

- `README.md`：项目介绍和当前状态。
- `docs/roadmap.md`：8 周 MVP 路线。
- `docs/architecture.md`：系统边界和 Feagine transfer。
- `docs/compatibility.md`：环境兼容性与已知例外。
- `docs/dependency_analysis.md`：三个仓库依赖和源码入口。
- `docs/phases/phase_00_environment.md`：Phase 0 完整过程。
- `docs/notes/week_01.md`：第一周记录。
- `docs/experiment_template.md`：实验记录模板。
- `experiments/act_baseline/README.md`：PushT ACT contract 和后续计划。
- `configs/env/evla-core.yml`：conda 基础环境。
- `configs/env/pusht_96.yaml`：PushT 96×96 runtime contract。
- `scripts/check_phase0_core.py`：无下载、无 GUI、无训练的 core smoke test。

## 11. 当前阶段结论

已完成的是 Phase 0 Core + PushT + ACT interface，不是完整 Phase 0。

已完成：

- CUDA / Blackwell / BF16；
- LeRobot training runtime；
- PushT headless simulator；
- LeRobot observation preprocessing；
- ACT image/state → action chunk interface；
- 环境配置、文档和重复 smoke script。

尚未完成：

- MuJoCo package 与 EGL；
- LeRobot `libero` extra / `hf-libero`；
- LIBERO reset/step/render；
- W&B login 与真实 run；
- TensorBoard 方案；
- 任何 dataset 下载；
- 任何 checkpoint 下载；
- 任何训练；
- 任何 GUI 仿真。

## 12. 下一步：Phase 0-D6

下一对话应从只读依赖分析开始，不直接安装：

1. 检查 pinned LeRobot `pyproject.toml` 中 `libero` extra 的准确依赖。
2. 检查 LeRobot LIBERO config、factory、processor 和 evaluation 入口。
3. 检查 `hf-libero`、MuJoCo、robosuite 与 Python 3.12 的当前约束。
4. 给用户一个 pip resolver dry-run 命令，让用户在宿主执行并回传完整摘要。
5. 评估是否会替换当前 torch/torchvision、NumPy、OpenCV 或其他核心包。
6. 估计安装量和磁盘增量。
7. 先给兼容性结论，再请求“安装最小 MuJoCo/LIBERO runtime”的单独授权。
8. 安装后只做 EGL reset、dummy step、RGB render，不启动 GUI，不下载示范数据。

Phase 0-D6 开始前不要：

- 安装原始 LIBERO `requirements.txt`；
- 执行原始 README 的旧 Python/PyTorch 安装命令；
- 修改当前 torch/torchvision；
- 自动下载 LIBERO demonstration dataset；
- 启动 MuJoCo viewer；
- 修改三个 submodule。

## 13. 新助手的工作方式

新的助手应：

1. 先完整读取本 handoff 和下列文件：
   - `docs/phases/phase_00_environment.md`
   - `docs/compatibility.md`
   - `docs/dependency_analysis.md`
   - `experiments/act_baseline/README.md`
2. 再检查 Git status、submodule commit 和当前环境，不要仅依赖摘要。
3. 对用户粘贴的每段输出：
   - 先判断 PASS / FAIL / expected exception；
   - 解释输出意味着什么；
   - 区分功能验证、性能 benchmark 和训练结果；
   - 将新事实加入阶段文档；
   - 再给下一条最小命令。
4. 每次只推进一个安全 gate。
5. 安装、下载、训练、GUI、上传和 commit 均需按用户既定方式单独确认。
6. 不虚构任何 success rate、loss、FPS、显存或简历结果。

## 14. 建议的新对话首条消息

如果新对话仍在同一个 Codex workspace，用户可以直接发送：

```text
请继续 Embodied VLA Lab 项目。

项目根目录：
/home/kxj/embodied_ws/embodied_vla_lab

请先完整读取：
/home/kxj/embodied_ws/embodied_vla_lab/docs/handoffs/2026-08-25_phase0_core_handoff.md

然后核对：
1. git status 和最近提交
2. 三个 submodule commit
3. docs/phases/phase_00_environment.md
4. docs/compatibility.md
5. docs/dependency_analysis.md

不要安装任何东西，不要下载数据或模型，不要训练，不要启动 GUI。

请先用中文总结你从 handoff 中恢复出的：
- 当前已完成阶段
- 已验证环境
- 两个已知例外
- 尚未完成事项
- Phase 0-D6 的执行计划

确认理解后，从 Phase 0-D6 的只读 MuJoCo/LIBERO 依赖分析开始。后续你给我命令，我在宿主终端执行并把完整运行结果发给你；你需要根据结果分析、记录并指导下一步。
```

如果是无法直接访问本地 workspace 的普通 ChatGPT 对话，应先上传本 Markdown 文件，再发送同样的提示，并把“请读取本地路径”改成“请完整读取我上传的 handoff 文件”。普通 ChatGPT 无法只凭路径读取用户电脑文件。
