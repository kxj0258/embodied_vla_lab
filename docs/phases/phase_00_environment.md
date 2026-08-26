# Phase 0 — Environment Preparation

## 实验目的

为 RTX 5070 Ti Laptop GPU 建立可复现、相互隔离的机器人学习环境。在引入 MuJoCo、LIBERO 和大模型前，先验证驱动、PyTorch、Blackwell CUDA 架构、LeRobot、PushT 和 ACT 的最小接口。

## 当前状态

状态：完成。

### 已完成

- [x] 收集宿主机 GPU、驱动、OS、conda 和工具链信息。
- [x] 固定 LeRobot、LIBERO、OpenVLA 三个上游 submodule commit。
- [x] 分析三个仓库的依赖边界与关键代码入口。
- [x] 创建 `evla-core`，安装 Python、FFmpeg 和 PyTorch cu128。
- [x] 验证 RTX 5070 Ti、compute capability 12.0、BF16 和 `sm_120`。
- [x] 安装 LeRobot 0.6.2 editable training 依赖。
- [x] 建立无 GUI 的 PushT reset/step/render 闭环。
- [x] 验证 LeRobot observation preprocessing 与 ACT image-to-action 接口。
- [x] 记录 OpenCV metadata 例外和 PushT 图像尺寸差异。
- [x] 完成 Phase 0-D6 首轮 MuJoCo/LIBERO resolver dry-run。
- [x] 按 curated 两阶段方案安装 MuJoCo/LIBERO Python runtime，保持唯一 headless OpenCV。
- [x] 验证 MuJoCo 3.8.1 的 NVIDIA EGL 离屏物理步进与 RGB render。
- [x] 完成 LIBERO reset、dummy step、双相机 observation 和离屏 RGB 检查。
- [x] 由宿主终端执行固化后的 `scripts/check_phase0_core.py`。
- [x] 决定并验证本地 TensorBoard event + JSONL 记录方案。

### 延后到后续阶段

- W&B login 和真实 run：进入 Phase 1 后按外部服务动作单独授权。
- dataset、checkpoint、训练、GUI、上传：均不属于 Phase 0 完成条件，继续分别授权。

## 宿主环境

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-138-generic |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| VRAM | 12227 MiB |
| Compute capability | 12.0 / `sm_120` |
| Driver | 580.173.02 |
| Driver CUDA ceiling | 13.0，不等于本地 Toolkit |
| RAM | 32 GB（用户提供） |
| Conda | 26.5.3 via Miniforge |
| Session | X11 |
| Free disk at initial audit | approximately 126 GB |

Codex 受限环境看不到 `/dev/nvidia*`；所有 GPU 验证均由用户在宿主终端执行并反馈原始输出。

## 固定上游版本

| Repository | Commit | Role |
|---|---|---|
| LeRobot | `8b256a6c0d4769c3cc3e7e98f04940126398a391` | dataset、policy、processor、train/eval |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | benchmark 语义与源码分析 |
| OpenVLA | `c8f03f48af692657d3060c19588038c7220e9af9` | VLA inference 与微调源码分析 |

详细依赖分析见 `docs/dependency_analysis.md`。

## evla-core 已验证版本

| Component | Verified version |
|---|---|
| Python | 3.12.14 |
| pip | 26.2.1 |
| FFmpeg | 7.1.1, conda-forge |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| LeRobot | 0.6.2, editable pinned submodule |
| gymnasium | 1.3.0 |
| gym-pusht | 0.1.6 |
| OpenCV | `opencv-python-headless` 4.13.0.92 |
| datasets | 4.8.5 |
| av | 15.1.0 |
| torchcodec | 0.11.1 |
| accelerate | 1.14.0 |
| W&B | 0.27.2 |

环境基础层记录在 `configs/env/evla-core.yml`。该文件不伪装成完整一键安装文件：PyTorch cu128 自定义索引、LeRobot editable 安装以及 PushT OpenCV workaround 仍按下面的分阶段命令执行。

## 执行与结果

### D1：conda 基础层

验证命令：

```bash
conda activate evla-core
which python
python --version
python -m pip --version
which ffmpeg
ffmpeg -version
```

结果：Python 3.12.14、pip 26.2.1、FFmpeg 7.1.1；环境初始大小约 750 MB。

### D2：PyTorch cu128

安装命令：

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

结果：

- `torch==2.11.0+cu128`。
- `torchvision==0.26.0+cu128`。
- `torch.version.cuda == "12.8"`。
- 安装后 `pip check` 无损坏依赖。
- 环境大小约 7.4 GB。

### D3：CUDA / Blackwell / BF16

实测结果：

| Check | Result |
|---|---|
| CUDA available | True |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Compute capability | `(12, 0)` |
| PyTorch arch list | includes `sm_120` |
| BF16 | supported |
| 4096×4096 BF16 matmul | finite, PASS |
| Matmul time | 0.103391 s，仅 smoke test |
| Peak PyTorch allocation | 184.12 MiB |

测试前空闲显存约 10.7 GB；测试结束后显存回落到桌面基线。该结果验证运行时兼容性，不作为模型训练吞吐指标。

### D4A：LeRobot resolver dry-run

直接解析 `.[training,pusht]` 会同时引入：

- LeRobot core 依赖的 `opencv-python-headless`。
- `gym-pusht` metadata 依赖的 `opencv-python`。

两者导出同一个 `cv2` 命名空间，因此没有直接接受该组合。

### D4B：LeRobot training 与 PushT runtime

LeRobot 安装：

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab/lerobot
python -m pip install -e ".[training]"
```

PushT 依赖采用分阶段安装：

```bash
python -m pip install \
  pygame==2.6.1 \
  pymunk==6.11.1 \
  shapely==2.1.2 \
  scikit-image==0.26.0

python -m pip install --no-deps gym-pusht==0.1.6
```

结果：

- 仅安装 `opencv-python-headless==4.13.0.92`，未安装 `opencv-python`。
- PushT reset、10 个随机 step 和 `rgb_array` render 均通过。
- 观测 keys：`agent_pos`、`pixels`。
- 图像观测：`uint8[96,96,3]`。
- state：`float64[2]`，LeRobot preprocessing 后为 `float32[1,2]`。
- action space：二维连续绝对目标位置，范围 `[0,512]`。
- visualization frame：`uint8[384,384,3]`。

`pip check` 的唯一结果为：

```text
gym-pusht 0.1.6 requires opencv-python, which is not installed.
```

这是已知 metadata 例外，不是缺少 `cv2`：实际 `cv2` import、物理步进和离屏渲染均已通过。禁止为了消除此提示而再安装非 headless OpenCV。

### D4C：LeRobot wrapper 与 ACT 接口

LeRobot observation preprocessing 已验证：

```text
pixels: uint8[B,H,W,C]
  → observation.image: float32[B,C,H,W], range [0,1]

agent_pos: float64[B,2]
  → observation.state: float32[B,2]
```

当前 pinned LeRobot 的默认 `PushtEnv` 声明 `observation.image=(3,384,384)`，但 `gym_kwargs` 未传递 observation width/height，`gym-pusht==0.1.6` 实际输出 `(3,96,96)`。本项目通过 `configs/env/pusht_96.yaml` 显式把 policy contract 固定为 96×96；暂不修改上游 submodule。

随机初始化 ACT 接口结果：

| Item | Result |
|---|---|
| Input image | `float32[1,3,96,96]` |
| Input state | `float32[1,2]` |
| Output action chunk | `float32[1,100,2]` |
| `chunk_size` | 100 |
| `n_action_steps` | 10 |
| Backbone | ResNet18, no pretrained download |
| Parameters | 51.59 M |
| First-pass latency | 0.309598 s |
| Peak PyTorch allocation | 222.56 MiB |
| Result | PASS |

延迟包含首次 CUDA/model warm-up；显存仅是随机初始化 FP32 inference。二者均不得当作训练性能或正式 benchmark。随机动作没有控制意义，未送入环境。

### D5：固化脚本宿主复核

2026-08-25 在宿主终端执行 `python scripts/check_phase0_core.py`，四个 gate 全部通过：

| Gate | Result |
|---|---|
| package contract | PASS |
| CUDA / BF16 | PASS |
| LeRobot PushT wrapper | PASS |
| ACT image-to-action | PASS |
| First-pass latency | 0.282970 s，非正式 benchmark |
| Peak PyTorch allocation | 223.43 MiB |

复核前 `nvidia-smi` 显示 GPU 已使用 4208 MiB、空闲 7540 MiB、利用率 58%。脚本仍通过，说明该检查所需显存余量较小；由于 GPU 非空闲，延迟不用于横向比较。

### D6：MuJoCo/LIBERO 首轮 resolver dry-run

2026-08-25 在宿主 `evla-core` 中执行：

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab/lerobot
conda activate evla-core
python -m pip install --dry-run --no-cache-dir --no-build-isolation -e ".[libero]"
```

结果分层判断：

| Gate | Result |
|---|---|
| Python | 3.12.14 |
| dependency resolver | PASS，完整解析并正常输出 `Would install` |
| planned packages | 59，包含 editable `lerobot` metadata refresh |
| torch / torchvision | 保持 2.11.0+cu128 / 0.26.0+cu128 |
| NumPy / SciPy | 保持 2.2.6 / 1.18.1 |
| MuJoCo | 解析为 3.8.1 CPython 3.12 Linux wheel |
| LIBERO runtime | `hf-libero==0.1.4`、`robosuite==1.4.0`、`robomimic==0.2.0` |
| Transformers | 受 LeRobot extra 约束解析为 5.5.4 |
| direct `.[libero]` install plan | FAIL / blocked：会加入 `opencv-python==5.0.0.93` |

这里的 FAIL 不是 resolver 失败，而是项目兼容性 gate 阻止直接安装。当前环境已经有
`opencv-python-headless==4.13.0.92`；两个 distribution 会写入同一个 `cv2` namespace，不能共存。
dry-run 没有安装任何包，也没有下载 dataset、模型或 LIBERO assets。

当前 LeRobot `uv.lock` 只能作为 pinned source 的参考解析；本次 pip 使用实时 package index，部分
传递依赖版本已高于 lock，例如 OpenCV、Numba、Hydra 和 TensorBoard。实际安装必须先形成显式
curated 集合并再次 dry-run，不能直接执行上面的非 dry-run 版本。

此外，`hf-libero==0.1.4` 会在首次运行时自动取得仿真 assets。EGL reset/render gate 前先检查
本地 asset cache；若不存在，下载 assets 需要单独确认，不能与 package 安装默认为同一次授权。

### D6：宿主静态 preflight

同日在宿主执行无安装、无 import MuJoCo 的只读检查，结果如下：

| Gate | Result |
|---|---|
| dry-run 后 package state | PASS；LIBERO、MuJoCo、robosuite、robomimic、Transformers 均未安装 |
| core package state | PASS；torch、torchvision、NumPy、headless OpenCV 均保持原版本 |
| `pip check` | 仅既有 `gym-pusht` OpenCV metadata 例外，expected exception |
| EGL loader | `libEGL.so.1` 可发现 |
| OpenGL / GLX | `libOpenGL.so.0` / `libGLX.so.0` 可发现 |
| NVIDIA EGL | `libnvidia-eglcore.so.580.173.02` 可发现 |
| GLVND vendor | NVIDIA 与 Mesa JSON 均存在，NVIDIA 指向 `libEGL_nvidia.so.0` |
| NVIDIA devices | `/dev/nvidia0`、ctl、modeset、uvm 和 uvm-tools 可见 |
| render env vars | 均未设置，后续 smoke 时临时设置 `MUJOCO_GL=egl` |
| LIBERO asset cache | 四个候选路径均不存在 |
| pip cache | 既存约 3.1 GiB，不删除，不计作本轮新增量 |
| `/home` free disk | 115 GiB |

静态 EGL preflight 判为 PASS，但不等于 MuJoCo EGL renderer 已验证。asset cache 缺失也不是功能
FAIL，而是后续 reset 前需要单独授权的下载 gate。

### D6：curated resolver dry-run

为避免标准 extra 引入第二个 `cv2` wheel，宿主执行了两阶段 dry-run：

1. 显式固定 core 版本并解析除 `hf-libero`、`robosuite` 外的 runtime dependencies；
2. 对 `robosuite==1.4.0` 和 `hf-libero==0.1.4` 单独使用 `--no-deps`。

结果：

| Gate | Result |
|---|---|
| curated dependency resolver | PASS |
| stage 1 planned packages | 55 |
| stage 2 planned packages | 2，严格为 `hf-libero` 和 `robosuite` |
| `opencv-python` in either stage | 未出现，PASS |
| core package changes | 无，PASS |
| environment mutation | 无，仍是 dry-run |

该两阶段方案判为有条件 PASS，可以进入单独的 package 安装授权 gate。安装后因为两个包的 metadata
不能把 headless distribution 识别为 `opencv-python`，预计 `pip check` 将有且仅有三条 OpenCV
例外：`gym-pusht`、`hf-libero`、`robosuite`。三条只能在 `cv2` import、PushT 回归和 LIBERO EGL
render 全部通过后接受。

参考 pinned lock 的 artifact 大小，排除约 68.7 MiB 的非 headless OpenCV 后，Python distributions
压缩下载量粗估约 0.33 GiB，其中 robosuite wheel 约 184.5 MiB；实时 index 版本略有漂移，实际
下载与落盘值需在安装时记录。LIBERO assets 不包含在该估计内。

### D6：Stage 1 首次安装尝试与构建例外

经单独授权后首次执行 Stage 1，pip 完成解析并成功构建多个 sdist，但在构建
`egl_probe==1.0.2` wheel 时失败：

```text
CMake Error at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```

根因是 `egl_probe` 的旧 CMake policy declaration 与当前 `cmake==4.1.3` 不兼容。前面的
setuptools package discovery warnings 不是终止原因。pip 在 wheel build 阶段中止，没有进入
install transaction；后检确认所有 Stage 1 新包均未安装，core pins 未变化，`pip check` 仍只有
PushT 一条已知例外，环境大小仍为 8.4 GiB，磁盘仍可用 115 GiB。因此此次判为 **FAIL with clean
environment**，不需要卸载或回滚。

CMake 4.0+ 提供 `CMAKE_POLICY_VERSION_MINIMUM` 环境变量用于在不修改第三方项目时初始化 policy
cache。使用当前 CMake 的 `/tmp` 最小复现表明：默认配置稳定复现同一错误；仅对进程设置
`CMAKE_POLICY_VERSION_MINIMUM=3.5` 后配置 PASS，并只产生非阻断弃用警告。下一次 Stage 1 在原
命令前临时设置该变量，不降级 CMake、不修改上游源码。

### D6：Stage 1 安装 PASS

使用进程级 CMake policy 变量重试后，`hf-egl-probe`、`egl_probe`、robomimic、BDDL 和 antlr4
sdist wheel 均构建成功，pip 完成安装事务：

| Gate | Result |
|---|---|
| installed distributions | 55 |
| MuJoCo | 3.8.1，已安装，尚未 import/EGL 验证 |
| robomimic | 0.2.0 |
| Transformers | 5.5.4 |
| TensorBoard | 2.21.0，由 robomimic 传递安装，记录方案尚未验证 |
| torch / torchvision | 保持 2.11.0+cu128 / 0.26.0+cu128 |
| NumPy / SciPy | 保持 2.2.6 / 1.18.1 |
| OpenCV | 仅 headless 4.13.0.92；非 headless distribution 未安装 |
| `hf-libero` / robosuite | 尚未安装，符合 Stage 1 边界 |
| `pip check` | 仅 PushT 一条已知 OpenCV 例外 |
| environment size | 8.4 GiB → 9.0 GiB，约增加 0.6 GiB |
| `/home` free disk | 粗粒度 115 GiB → 114 GiB |

Stage 1 判为 PASS。成功安装只证明 package transaction 与 metadata 后检通过，不把 MuJoCo import、
EGL renderer、TensorBoard logging 或 simulator runtime 标记为 PASS。安装过程未下载 LIBERO assets。

### D6：Stage 2 安装 PASS

第二阶段使用 `--no-deps` 安装两个顶层 packages：

```bash
python -m pip install --no-cache-dir --no-deps \
  "robosuite==1.4.0" "hf-libero==0.1.4"
```

| Gate | Result |
|---|---|
| robosuite | 1.4.0，PASS |
| `hf-libero` | 0.1.4，PASS |
| actual downloads | 193.5 MB + 3.2 MB = 196.7 MB |
| non-headless OpenCV | 未安装，PASS |
| core pins | 全部保持，PASS |
| `pip check` | 恰好三条 OpenCV metadata 例外，无第四条 |
| environment size | 9.0 GiB → 9.6 GiB |
| total D6 package increment | 8.4 GiB → 9.6 GiB，约 1.2 GiB |
| LIBERO assets | 未下载 |

三条当前 metadata 例外分别来自 `gym-pusht`、robosuite 和 `hf-libero`。Stage 2 判为 package
installation PASS；只有后续 `cv2`/PushT 回归与 LIBERO EGL render 均通过后，才能把新增两条正式
归类为接受的预期例外。

### D6：OpenCV 与 Phase 0 core 回归

Stage 2 后重新执行 headless OpenCV 检查和 `scripts/check_phase0_core.py`：

| Gate | Result |
|---|---|
| pre-test GPU | 869 MiB used / 10879 MiB free / 9% utilization |
| `opencv-python` distribution | NOT INSTALLED |
| headless OpenCV | 4.13.0.92 |
| `cv2` runtime | 4.13.0，来自 `evla-core/site-packages/cv2` |
| OpenCV GUI backend | `NONE` |
| package contract | PASS |
| CUDA / BF16 | PASS |
| PushT declared/runtime | `(3,96,96)` / `(3,96,96)`，PASS |
| ACT image-to-action | PASS |
| `pip check` | 恰好三条预期 OpenCV metadata 例外 |

本次 ACT first-pass latency 为 0.335201 s，peak PyTorch allocation 为 223.43 MiB。前者仍包含首次
warm-up，只作 smoke 记录，不作为性能 benchmark。该回归证明两阶段安装没有破坏 core、PushT 或
唯一 headless `cv2` provider；robosuite/LIBERO 对该 provider 的运行时兼容仍待后续 gate。

### D6：独立 MuJoCo EGL smoke PASS

在不 import LIBERO、不使用任何外部 asset 的内联最小 XML 上，以命令级
`MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0` 执行物理步进和离屏渲染：

| Gate | Result |
|---|---|
| MuJoCo | 3.8.1，import PASS |
| EGL device | NVIDIA device 0，context/render PASS |
| simulation | `time 0.0 → 0.1`，自由球体 z `1.0 → 0.946045`，PASS |
| RGB | `(120,160,3)`、`uint8`、range `0..255`、finite，PASS |
| RGB content | mean 35.806111、std 30.579974，非恒定图像 |
| pre-test GPU | 859 MiB used / 10889 MiB free / 22% utilization |
| LIBERO asset cache | 四个候选路径仍全部不存在 |

该 gate 将 MuJoCo import、物理执行、NVIDIA EGL context 和无窗口 RGB 输出整体判为 PASS。GPU
利用率是测试前瞬时值，不作为性能指标。资产目录的后检证明独立 smoke 没有隐式下载 LIBERO 数据；
它不验证 robosuite 或 LIBERO simulator，二者继续留在下一 gate。

### D6：LIBERO import boundary 静态检查

对已安装 wheel 源码做只读检查后确认：顶层 `libero/__init__.py` 为空；导入
`libero.libero` 会在 `LIBERO_CONFIG_PATH` 下初始化配置，但不会调用资产下载。自动下载入口
`get_assets_path()` 只在 LIBERO environment/object 实例化时执行，并通过
`snapshot_download(repo_id="lerobot/libero-assets", repo_type="dataset")` 写入默认
`~/.cache/libero/assets`。robosuite 的下载 helper 也只在脚本作为 `__main__` 运行时调用，普通 import
不会调用；robosuite wheel 已自带约 586 MiB 的 `models/assets`。

因此下一条 import gate 使用一次性 `LIBERO_CONFIG_PATH`、预写配置和 `HF_HUB_OFFLINE=1`，只导入
benchmark/env/wrapper 模块及实例化不接触 simulator 的 task suite，明确不实例化
`OffScreenRenderEnv`/`LiberoEnv`，并在结束后复核四个用户目录。

### D6：LIBERO 无下载 import boundary PASS

宿主按上述边界实际执行后的结果：

| Gate | Result |
|---|---|
| versions | MuJoCo 3.8.1、robosuite 1.4.0、`hf-libero` 0.1.4 |
| OpenCV | headless 4.13.0.92，runtime 4.13.0，`GUI: NONE` |
| robosuite import | PASS；wheel 内 `models/assets` 存在 |
| LIBERO env import | `OffScreenRenderEnv` PASS |
| LeRobot wrapper import | `lerobot.envs.libero.LiberoEnv` PASS |
| action contract | 7D、`[-1,1]`；dummy action `[0,0,0,0,0,0,-1]` |
| benchmark metadata | `libero_spatial` 10 tasks，task 0 language/name 可读 |
| BDDL / init state | task 0 两个本地文件均存在 |
| user paths | 四个候选路径前后均不存在 |

robosuite import 输出三条缺少 `macros_private.py` 的提示。该文件用于用户覆盖宏；当前默认宏已成功
加载，import contract 不依赖运行提示中的 setup 脚本，因此归类为预期警告，不执行该脚本。此 gate
证明 Python 3.12 模块、benchmark 元数据、LeRobot wrapper 和唯一 headless `cv2` 的导入边界正常；
它仍未实例化 simulator，所以不提前接受 LIBERO render gate 或新增两条 OpenCV metadata 例外。

### D6：LIBERO asset 下载预算

2026-08-25 只读核对 Hugging Face 官方 dataset repo `lerobot/libero-assets`：网页显示 main 总体积
422 MB，当前 verified commit 为 `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`。内容限于
`articulated_objects`、`scenes`、`stable_hope_objects`、`stable_scanned_objects`、`textures`、
`turbosquid_objects` 及少量顶层 XML/STL，不是 demonstrations、训练 dataset 或模型。

计划固定该 revision 下载至 `~/.cache/libero/assets`，传输/落盘按约 0.5 GiB 估计并预留 1 GiB
安全预算；当前 `/home` 约 114 GiB 可用，空间不是阻断项。该下载尚未执行，必须获得与 Python
package 安装分离的明确资产下载授权。

### D6：LIBERO asset 首次下载尝试 FAIL（环境干净）

获得单独授权后，以固定 revision 调用 `snapshot_download`。调用在 httpx client 初始化阶段失败：

```text
ValueError: Unknown scheme for proxy URL URL('socks://127.0.0.1:7897/')
```

该错误发生在 Hugging Face repo tree 请求之前，根因是宿主代理 URL scheme 不受当前 httpx 接受，
不是 repo、revision、磁盘或认证失败。下载目标未创建，两个 Hub repo cache 和 `~/.libero` 仍不存在；
`du` 的“没有那个文件或目录”是主失败的预期连带结果。粗粒度磁盘仍为 113 GiB free。因此判为
download FAIL / clean recovery，不清理任何文件，也不重复盲试。下一 gate 先脱敏核对 proxy 环境变量
和已安装 SOCKS transport 支持，再选择命令级代理修正或直连测试。

### D6：代理诊断 PASS

脱敏、无网络检查确认大小写代理变量成对存在：`HTTP_PROXY`/`HTTPS_PROXY` 与小写版本均为
`http://127.0.0.1:7897`，而 `ALL_PROXY`/`all_proxy` 为 `socks://127.0.0.1:7897`，均无
credentials。httpx 0.28.1 client construction 实测：`socks://` 因 unknown scheme FAIL，
`socks5://` 因未安装 `socksio` FAIL，`http://` PASS；环境也没有 PySocks。

根因因此确定为 httpx 初始化完整环境代理映射时被无效 ALL proxy 条目阻断。无需为本项目安装
SOCKS transport，也不改永久 shell 配置；后续只在 Hugging Face 命令前使用
`env -u ALL_PROXY -u all_proxy`，保留已有 HTTP/HTTPS proxy。下一 gate 先访问固定 revision 的
repo metadata，成功后才恢复 422 MB snapshot 下载。

### D6：固定 asset revision 网络 metadata gate PASS

在上述命令级代理边界下，`HfApi.repo_info` 成功访问公共 dataset repo：返回 ID
`lerobot/libero-assets`，resolved SHA 与请求的
`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` 完全一致，并报告 586 个 sibling entries。
请求前后四个用户路径均不存在，确认该 gate 只读取 metadata、未下载 snapshot。HTTP proxy 路径和
固定 revision 均已验证，下一步恢复已授权的 asset 下载；继续使用命令级变量，不修改永久代理设置。

### D6：固定 LIBERO simulator assets 下载 PASS

在命令级取消 `ALL_PROXY`/`all_proxy`、保留 HTTP/HTTPS proxy 后，固定 revision
`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` 的 snapshot 成功下载：

| Gate | Result |
|---|---|
| fetched payload | 586 / 586 files，PASS |
| payload bytes | 422,320,936 bytes / 402.76 MiB |
| local disk usage | 409 MiB |
| HF local metadata | 1,175 files / 179,955 bytes |
| required directories | articulated、scenes、HOPE、scanned、textures、TurboSquid 全部存在 |
| destination | `~/.cache/libero/assets` |
| extra Hub repo cache | models/datasets 两个候选路径均不存在 |
| `~/.libero` | 不存在 |
| `/home` free | coarse value 保持 113 GiB |

公共仓库未认证提示只影响 rate limit，不影响本次完整性，归为预期提示。该 gate 只取得 simulator
assets，没有 demonstrations、训练 dataset 或模型。下载授权目标已完成；后续 simulator gate 设
`HF_HUB_OFFLINE=1`，不得再发起网络下载。

### D6：LeRobot LIBERO reset / dummy step / EGL RGB PASS

固定 assets 下载后，在 `HF_HUB_OFFLINE=1`、`MUJOCO_GL=egl`、NVIDIA device 0 下，以临时
`LIBERO_CONFIG_PATH` 实例化 LeRobot `LiberoEnv`。测试使用 `libero_spatial` task 0、单环境、
128×128 双相机、20 Hz relative control、固定 init state、hard reset 和 10 次 settle steps：

| Gate | Result |
|---|---|
| deferred construction | reset 前底层 simulator 为 `None`，PASS |
| reset | simulator 创建成功，`is_success=False`，PASS |
| agent view | `(128,128,3)` `uint8`，range `6..230`，finite/non-constant |
| wrist view | `(128,128,3)` `uint8`，range `3..243`，finite/non-constant |
| robot state | eef pose/matrix、gripper qpos/qvel、joint pos/vel shape 与 finite 全部 PASS |
| observation contract | reset 和 step 均被 Gym observation space 接受 |
| action contract | 7D float32 `[0,0,0,0,0,0,-1]` 被 action space 接受 |
| dummy step | reward 0.0，未 terminated/truncated，PASS |
| `render()` | `(128,128,3)` `uint8`，range `6..230`，finite/non-constant |
| cleanup | environment close 与临时配置删除 PASS |
| offline boundary | asset manifest 前后均为 586 files / 422,320,936 bytes；`~/.libero` 不存在 |
| GPU snapshots | before 1069 MiB / after 1026 MiB used；只作边界记录，不是峰值 |

“Local assets not found / Assets already downloaded”是 `hf-libero` 先查 wheel 内目录、再命中固定
cache 的组合提示；manifest 不变且 offline mode 生效，未发生联网下载。robosuite 的 private macro
提示仍为预期警告。该 gate 证明 robosuite/LIBERO image path 可由唯一的 headless `cv2` provider
满足，因此 `gym-pusht`、robosuite、`hf-libero` 三条 `opencv-python` metadata 报告正式接受为
有意保留的预期例外；不得为消除它们安装非 headless OpenCV。

### D6：最终状态审计 PASS

完成 simulator gate 后执行只读审计：

| Gate | Result |
|---|---|
| Git | `main` / HEAD `1a9a5b2`；仅三份阶段文档修改 |
| diff hygiene | `git diff --check` 无输出，PASS |
| LeRobot submodule | `8b256a6c0d4769c3cc3e7e98f04940126398a391` |
| LIBERO submodule | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| OpenVLA submodule | `c8f03f48af692657d3060c19588038c7220e9af9` |
| core | torch 2.11.0+cu128、torchvision 0.26.0+cu128、NumPy 2.2.6、SciPy 1.18.1 |
| simulator | MuJoCo 3.8.1、robosuite 1.4.0、robomimic 0.2.0、`hf-libero` 0.1.4 |
| other D6 runtime | Transformers 5.5.4、TensorBoard 2.21.0 |
| OpenCV | headless 4.13.0.92 only；runtime 4.13.0，`GUI: NONE` |
| assets | 586 files / 422,320,936 bytes；`~/.libero` 不存在 |
| storage | environment 9.6 GiB；assets 409 MiB；`/home` 113 GiB free |
| `pip check` | 恰好三条已接受 OpenCV metadata 例外，无第四条 |

Phase 0-D6 至此关闭。该结论是依赖、离屏 simulator 与接口验证，不是训练结果、任务成功率或性能
benchmark。文档改动尚未 commit；commit 继续需要单独授权。

### 本地日志方案 PASS

TensorBoard 2.21.0 由 robomimic runtime 传递安装，但 LeRobot 当前源码没有原生
`SummaryWriter` 路径。Phase 0 因此验证日志 primitive，不在尚未运行的 trainer 中强行加入 callback：

| Gate | Result |
|---|---|
| TensorBoard write | 单个 event file，296 bytes |
| scalar tags | `smoke/value`、`smoke/learning_rate` |
| steps | 两个 tag 均精确回读 `[0,1]` |
| values | value 精确回读；learning rate 为预期 float32 尾差，`allclose` PASS |
| JSONL | 两条结构化记录逐行写入并完全回读 |
| cleanup | 临时目录删除，Git 未新增文件 |
| server / GUI | 未启动 |

后续实验采用忽略版本控制的 `outputs/<experiment_id>/`：TensorBoard events 写入
`tensorboard/`，同一步数和 tag 同步写入 `metrics.jsonl`，并保存 resolved config/environment
manifest；可追踪的结论继续写入实验 Markdown 记录。`.gitignore` 已覆盖 `outputs/`、`tensorboard/`、
`runs/` 和 event files。Phase 1 首次训练前再实现或选择最小 metric sink；W&B 作为 LeRobot 原生
外部追踪器，登录和真实 run 继续单独授权。

## 固化文件

- `configs/env/evla-core.yml`：conda 管理的最小基础层。
- `configs/env/pusht_96.yaml`：PushT 96×96 observation/action runtime contract。
- `scripts/check_phase0_core.py`：宿主侧可重复执行的 core smoke test。

## Phase 1C environment addendum（2026-08-26）

经单独授权在`evla-core`安装`diffusers==0.39.0`，与pinned LeRobot的约束
`diffusers>=0.38.0,<0.40.0`及其lockfile版本一致。严格离线Diffusion配置preflight、2-step GPU资源
gate和30k正式训练均PASS。该后续增量没有合并进conda base YAML；原因与PyTorch自定义index、editable
LeRobot和`gym-pusht --no-deps`相同：它属于按阶段授权的pip层。完整运行入口见
`scripts/runs/pusht.sh`，结果见`results/phase1c_diffusion_pusht_*.yaml`。

## 下一步

1. 单独授权后提交 Phase 0 文档记录。
2. 规划 Phase 1 ACT baseline；dataset 下载、训练、W&B 和上传继续分别授权。

## 思考

- 驱动上限、PyTorch wheel runtime 和 compute capability 必须分开记录；`nvidia-smi` 的 CUDA 13.0 不意味着应安装 CUDA 13 Toolkit。
- 环境 feature declaration 必须与 simulator 实际 tensor 对照。只读取配置会漏掉当前 PushT 384/96 的不一致。
- 包管理器的 metadata 完整性和运行时接口完整性不是同一件事。OpenCV workaround 必须保留原因、实测证据和禁止操作。
- ACT 的 action chunk shape 只说明接口正确；没有 dataset statistics、训练权重和 postprocessor 时，动作数值不可解释。
