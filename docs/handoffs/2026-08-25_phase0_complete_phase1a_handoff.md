# Embodied VLA Lab — Phase 0 Complete / Phase 1A Handoff

更新时间：2026-08-25

用途：供新的 Codex/ChatGPT 对话恢复本轮工作，并从 Phase 1A 继续。
项目根目录：`/home/kxj/embodied_ws/embodied_vla_lab`

## 1. 新对话必须先知道的结论

Phase 0 已完成并提交。当前正式提交：

```text
4d67e21 docs: complete phase 0 environment validation
```

Phase 0 已验证：

- RTX 5070 Ti Laptop GPU / Blackwell `sm_120` / BF16；
- PyTorch 2.11.0+cu128 与 torchvision 0.26.0+cu128；
- LeRobot 0.6.2 editable runtime；
- PushT 96×96 headless reset/step/render；
- ACT 随机初始化 image/state → action chunk 接口；
- MuJoCo 3.8.1 NVIDIA EGL 离屏 physics/render；
- curated MuJoCo/LIBERO Python runtime；
- 固定 revision 的 LIBERO simulator assets；
- LeRobot LIBERO offline reset/dummy-step/双相机 EGL RGB；
- TensorBoard event + JSONL 本地日志 primitive。

Phase 1A 尚未开始执行。上一对话给出的 `lerobot/pusht` metadata preflight **用户暂未执行**。
因此当前没有 PushT dataset snapshot，没有训练，没有 checkpoint，没有 rollout 结果。

## 2. 用户协作与授权边界

- 中文交流。
- 助手先解释 gate，再给命令；用户在宿主终端执行并粘贴完整输出。
- 对每段输出先判定 `PASS`、`FAIL` 或预期例外，再解释、更新项目记录、给下一条最小命令。
- 用户要求宿主执行命令；不要代替用户运行安装、下载、GPU、训练或 GUI 命令。
- 安装、dataset/model 下载、训练、GUI、W&B/上传、Git commit 均需分别明确授权。
- 不自动下载 demonstrations、模型或大型 checkpoint。
- 不自动启动 MuJoCo viewer、TensorBoard server 或其他 GUI。
- 不修改三个 submodule。
- 不虚构 loss、success rate、FPS、峰值显存、wall time 或简历结果。
- 当前任务不使用多 agent。

## 3. Git 与 submodule 固定状态

Phase 0 提交验收时：

```text
4d67e21 (HEAD -> main) docs: complete phase 0 environment validation
1a9a5b2 docs: add phase 0 core handoff
e88d63f chore: lock phase 0 core environment
```

三个 submodule：

| Submodule | Commit |
|---|---|
| LeRobot | `8b256a6c0d4769c3cc3e7e98f04940126398a391` |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| OpenVLA | `c8f03f48af692657d3060c19588038c7220e9af9` |

提交 `4d67e21` 后工作树曾验证为干净。本 handoff 是该提交之后新增的本地文件，尚未提交；新对话开始
时应预期 `git status --short` 只显示：

```text
?? docs/handoffs/2026-08-25_phase0_complete_phase1a_handoff.md
```

不要把该未跟踪 handoff 当成异常，也不要未经授权提交。

## 4. 已验证宿主与核心环境

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-138-generic |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| VRAM | 12227 MiB |
| Driver | 580.173.02 |
| Compute capability | `(12,0)` / `sm_120` |
| Conda env | `evla-core` |
| Python | 3.12.14 |
| pip | 26.2.1 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| NumPy | 2.2.6 |
| SciPy | 1.18.1 |
| Gymnasium | 1.3.0 |
| gym-pusht | 0.1.6 |
| LeRobot | 0.6.2 editable |
| OpenCV | `opencv-python-headless==4.13.0.92` only |
| MuJoCo | 3.8.1 |
| robosuite | 1.4.0 |
| robomimic | 0.2.0 |
| `hf-libero` | 0.1.4 |
| Transformers | 5.5.4 |
| TensorBoard | 2.21.0 |

最终审计时：

- conda environment：9.6 GiB；
- LIBERO assets：409 MiB；
- `/home` 可用空间：113 GiB（`df` 粗粒度）；
- OpenCV runtime：4.13.0，`GUI: NONE`；
- `opencv-python` 非 headless distribution：未安装。

## 5. Phase 0 Core 与 PushT/ACT 证据

固化脚本：`scripts/check_phase0_core.py`。

四个 gate 全部 PASS：

1. package contract；
2. CUDA / BF16 / `sm_120`；
3. LeRobot PushT wrapper；
4. ACT image-to-action interface。

PushT 项目 contract：`configs/env/pusht_96.yaml`。

| Contract | Value |
|---|---|
| raw image | HWC `(96,96,3)` uint8 |
| policy image | CHW `(3,96,96)` float32 |
| raw agent state | `(2,)` float64 |
| policy state | `(2,)` float32 |
| action | 2D absolute target `[x,y]` |
| environment action range | `[0,512]` |
| control rate | 10 Hz |
| ACT smoke chunk | `(1,100,2)` float32 |
| ACT parameters | 51.59 M |

ACT latency numbers包含首次 warm-up，不是 benchmark；模型随机初始化，动作无控制意义，未送入环境。

## 6. 两个核心已知例外

### 6.1 OpenCV metadata 例外

项目有意只保留 headless wheel，避免两个 distributions 争用同一 `cv2` namespace。当前
`python -m pip check` 预期恰好三条：

```text
gym-pusht 0.1.6 requires opencv-python, which is not installed.
robosuite 1.4.0 requires opencv-python, which is not installed.
hf-libero 0.1.4 requires opencv-python, which is not installed.
```

三条均已有运行时证据：PushT 与 LIBERO reset/step/EGL RGB 全部 PASS。不要为了消除提示安装
`opencv-python`。如果出现第四条 `pip check` 错误，必须单独分析。

### 6.2 PushT 384/96 declaration mismatch

Pinned LeRobot `PushtEnv` 默认声明 `(3,384,384)`，但 `gym-pusht==0.1.6` runtime 实际为
`(3,96,96)`。项目通过 `configs/env/pusht_96.yaml` 显式对齐，不修改上游 submodule。

## 7. Phase 0-D6 安装与构建结论

标准 `pip install -e ".[libero]"` 的 resolver 可解，但会安装非 headless
`opencv-python==5.0.0.93`，因此项目判为 compatibility FAIL，继续禁止。

实际采用两阶段 curated 安装：

1. 正常解析并安装 55 个缺失 dependencies，同时固定 core packages 和 headless OpenCV；
2. `--no-deps` 安装 `robosuite==1.4.0` 与 `hf-libero==0.1.4`。

Stage 1 首次构建 `egl_probe==1.0.2` 时，CMake 4.1.3 拒绝旧 policy minimum。环境保持干净。使用
官方支持的进程级：

```text
CMAKE_POLICY_VERSION_MINIMUM=3.5
```

重试后 PASS；没有降级 CMake，没有 patch 上游源码。

两阶段结果：

- 环境由 8.4 GiB 增至 9.6 GiB；
- core torch/torchvision/NumPy/SciPy 未漂移；
- 非 headless OpenCV 未安装；
- Python package 安装阶段没有下载 LIBERO assets。

## 8. MuJoCo/LIBERO 运行时证据

### 8.1 独立 MuJoCo EGL

命令级环境：

```text
MUJOCO_GL=egl
MUJOCO_EGL_DEVICE_ID=0
```

最小内联 XML physics 从 0.0 s 前进至 0.1 s；RGB 为 `(120,160,3)` `uint8`、range `0..255`、
finite 且非恒定。PASS。

### 8.2 LIBERO assets

固定 Hugging Face dataset repo：

```text
repo: lerobot/libero-assets
revision: 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6
target: /home/kxj/.cache/libero/assets
```

最终 asset contract：

```text
payload files: 586
payload bytes: 422320936
du: 409M
```

六个目录存在：articulated objects、scenes、stable HOPE objects、stable scanned objects、textures、
TurboSquid objects。没有 demonstrations、训练 dataset 或模型。

### 8.3 Hugging Face proxy workaround

宿主代理：

```text
HTTP_PROXY/http_proxy   = http://127.0.0.1:7897
HTTPS_PROXY/https_proxy = http://127.0.0.1:7897
ALL_PROXY/all_proxy     = socks://127.0.0.1:7897
```

httpx 0.28.1 会因 `socks://` unknown scheme 在 client 初始化时失败；`socks5://` 又需要未安装的
`socksio`。不安装新包、不修改永久 shell。对 Hugging Face 命令使用：

```bash
env -u ALL_PROXY -u all_proxy <command>
```

保留现有 HTTP/HTTPS proxy。该 workaround 已通过 pinned repo metadata 和 422 MB asset snapshot。

### 8.4 LeRobot LIBERO offline gate

严格设置 `HF_HUB_OFFLINE=1`，使用临时 `LIBERO_CONFIG_PATH`；`libero_spatial` task 0：

- deferred simulator construction PASS；
- hard reset、固定 init state、10 settle steps PASS；
- 7D dummy action `[0,0,0,0,0,0,-1]` step PASS；
- agent/wrist 两路 `(128,128,3)` `uint8` RGB 非恒定且 finite；
- eef、gripper、joint state shape/finite PASS；
- reset/step observation 与 action 均满足 Gym spaces；
- LeRobot `render()` PASS；
- environment close、临时 config cleanup PASS；
- asset manifest 前后保持 586 / 422320936；
- `~/.libero` 前后均不存在。

robosuite 缺少 `macros_private.py` 的提示是可选用户宏警告，不运行其 setup 脚本。
“Local assets not found / Assets already downloaded”是先查 wheel 内目录、再命中固定 cache 的组合提示，
不是重新下载。

## 9. 本地日志决策

TensorBoard primitive smoke PASS：

- `SummaryWriter` 生成一个 event file；
- `EventAccumulator` 回读 `smoke/value` 与 `smoke/learning_rate`；
- steps `[0,1]` 正确；
- learning-rate 尾差为 event scalar float32 表示，`allclose` PASS；
- JSONL 两条记录完整 roundtrip；
- 临时目录删除；
- 未启动 TensorBoard server 或 GUI。

Phase 1 本地布局：

```text
outputs/<experiment_id>/
  tensorboard/
  metrics.jsonl
  resolved_config.*
  environment.*
  checkpoint_manifest.*
```

`outputs/`、`tensorboard/`、`runs/` 和 event files 已在 `.gitignore`。可追踪结论写入实验 Markdown。
LeRobot 当前源码没有原生 TensorBoard writer；Phase 1 首次训练前再实现或选择最小 metric sink。
W&B login/真实 run 是外部动作，继续单独授权。

## 10. 当前未完成事项

- Phase 1A `lerobot/pusht` metadata preflight 尚未执行；
- 未固定 PushT dataset revision；
- 未评估 PushT dataset 总大小与本地缓存；
- 未下载 PushT dataset；
- 未检查 dataset 单样本、episode、statistics、normalization、padding mask；
- 未决定 ACT `chunk_size`、`n_action_steps`、episode split；
- 未授权 pretrained ResNet18 下载；
- 未创建正式训练配置；
- 未训练、未生成 checkpoint、未 rollout；
- 未登录 W&B、未上传；
- 未启动任何 GUI。

用户说会等训练结束后在新对话继续；本项目在本轮没有启动训练。若用户指的是其他进程，新助手仍应
先检查 GPU/磁盘状态，不应把外部训练结果误记为本项目结果。

## 11. Phase 1A 只读起步计划

1. 查询官方 `lerobot/pusht` 当前完整 SHA、文件数、逻辑大小与顶层目录构成；
2. 检查 `$HF_LEROBOT_HOME` 与 revision-safe Hub cache 是否已有数据；
3. 只读检查 pinned LeRobot Dataset v3、ACT delta indices、padding 和 normalization 源码；
4. 形成 dataset revision、下载量、落盘位置和是否需要 ResNet18 权重的结论；
5. 单独请求 PushT dataset 下载授权；
6. 下载后先做单 episode/sample/statistics contract，不训练；
7. 形成极小 overfit/smoke 配置，再单独请求训练授权。

## 12. 下一条尚未执行的宿主命令

该命令只查询 Hub metadata，不下载 dataset snapshot：

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab
conda activate evla-core

env -u ALL_PROXY -u all_proxy \
HF_HUB_DISABLE_TELEMETRY=1 \
python - <<'PY'
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.constants import HF_HOME
from lerobot.utils.constants import HF_LEROBOT_HOME, HF_LEROBOT_HUB_CACHE

repo_id = "lerobot/pusht"

cache_paths = [
    HF_LEROBOT_HOME / repo_id,
    HF_LEROBOT_HUB_CACHE / "datasets--lerobot--pusht",
    Path(HF_HOME) / "hub/datasets--lerobot--pusht",
]

print("CACHE_BOUNDARY")
print(f"HF_HOME: {HF_HOME}")
print(f"HF_LEROBOT_HOME: {HF_LEROBOT_HOME}")
print(f"HF_LEROBOT_HUB_CACHE: {HF_LEROBOT_HUB_CACHE}")

before = {}
for path in cache_paths:
    before[path] = path.exists()
    print(f"{path}: exists={before[path]}")

api = HfApi()
info = api.repo_info(
    repo_id=repo_id,
    repo_type="dataset",
    revision="main",
    files_metadata=True,
)

siblings = list(info.siblings or [])
known_sizes = [(item.rfilename, item.size) for item in siblings if item.size is not None]
missing_sizes = [item.rfilename for item in siblings if item.size is None]
total_bytes = sum(size for _, size in known_sizes)

groups = defaultdict(lambda: {"files": 0, "bytes": 0})
for filename, size in known_sizes:
    top_level = filename.split("/", 1)[0]
    groups[top_level]["files"] += 1
    groups[top_level]["bytes"] += size

print("\nREMOTE_REPOSITORY")
print(f"repo id: {info.id}")
print(f"resolved sha: {info.sha}")
print(f"last modified: {info.last_modified}")
print(f"file entries: {len(siblings)}")
print(f"entries with known size: {len(known_sizes)}")
print(f"entries without size: {len(missing_sizes)}")
print(f"known logical bytes: {total_bytes}")
print(f"known logical MiB: {total_bytes / 1024**2:.2f}")
print(f"known logical GiB: {total_bytes / 1024**3:.4f}")

print("\nTOP_LEVEL_SIZE_BREAKDOWN")
for name in sorted(groups):
    values = groups[name]
    print(
        f"{name}: files={values['files']}, "
        f"bytes={values['bytes']}, "
        f"MiB={values['bytes'] / 1024**2:.2f}"
    )

print("\nLARGEST_FILES")
for filename, size in sorted(known_sizes, key=lambda item: item[1], reverse=True)[:15]:
    print(f"{size:>12}  {filename}")

if missing_sizes:
    print("\nMISSING_SIZE_ENTRIES")
    for filename in missing_sizes[:20]:
        print(filename)

print("\nPOST_METADATA_CACHE_BOUNDARY")
after = {}
for path in cache_paths:
    after[path] = path.exists()
    print(f"{path}: exists={after[path]}")

assert info.id == repo_id
assert info.sha
assert siblings
assert before == after

print("PushT repository metadata preflight: PASS")
print("No dataset snapshot was downloaded.")
PY

df -h /home/kxj
git status --short --branch
```

新助手收到输出后，应先判断 PASS/FAIL，记录完整 SHA、逻辑大小、cache side effects 和磁盘预算，
再决定是否向用户请求 dataset 下载授权。

## 13. 新对话开场检查

新助手必须先完整读取：

1. 本文件；
2. `docs/phases/phase_00_environment.md`；
3. `docs/compatibility.md`；
4. `docs/dependency_analysis.md`；
5. `experiments/act_baseline/README.md`；
6. `configs/env/pusht_96.yaml`。

然后只读核对：

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab
git status --short --branch
git log -3 --oneline --decorate
git submodule status --recursive
```

新助手应先用中文总结恢复出的 Phase 0 结论、三个 metadata 例外、PushT 96×96 例外、代理 workaround、
当前未完成项和 Phase 1A 计划。确认理解后，等待用户粘贴 metadata preflight 输出；不要误称该命令
已经执行。
