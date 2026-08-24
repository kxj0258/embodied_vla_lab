# Upstream Dependency and Code Analysis

更新时间：2026-08-25。

本分析基于项目 submodule 中实际固定的代码，而不是只参考在线 README。

## 1. 固定版本

| Repository | Commit | Commit date | Branch at clone |
|---|---|---|---|
| LeRobot | `8b256a6c0d4769c3cc3e7e98f04940126398a391` | 2026-08-24 | `main` |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | 2025-03-15 | `master` |
| OpenVLA | `c8f03f48af692657d3060c19588038c7220e9af9` | 2025-03-23 | `main` |

三个仓库均采用 shallow submodule。父仓库记录精确 gitlink，因此后续更新不会自动改变实验版本。

## 2. LeRobot

### 当前依赖事实

固定版本为 LeRobot `0.6.2`：

| Component | Constraint |
|---|---|
| Python | `>=3.12` |
| PyTorch | `>=2.7,<2.12.0` |
| torchvision | `>=0.22.0,<0.27.0` |
| NumPy | `>=2.0,<2.3` |
| Transformers extra | `>=5.4,<5.6` |
| Diffusers extra | `>=0.38,<0.40` |
| W&B | `>=0.24,<0.28` |
| CUDA index in source config | PyTorch cu128 |

源码安装文档明确说明 cu128 driver floor 为 570.86，并覆盖 Blackwell。宿主驱动为 580.173.02，满足该约束。

### 可选 extras

- `training`：dataset、accelerate、W&B。
- `pusht`：dataset、gym-pusht、pymunk。
- `diffusion`：diffusers。
- `libero`：dataset、Transformers、SciPy、`hf-libero>=0.1.4,<0.2.0`。

不使用 `.[all]`，因为它会安装大量无关机器人硬件、VLA、仿真和开发依赖，增加 resolver 冲突和磁盘占用。

### 分阶段安装策略

1. Phase 0/1A 只安装 `training,pusht`。
2. PushT ACT smoke test 通过后添加 `diffusion`。
3. Phase 2 前再添加 `libero`。

这样能区分 LeRobot core、Diffusion Policy 和 LIBERO 三层问题。

### 关键代码入口

#### Dataset

- `src/lerobot/datasets/lerobot_dataset.py`：主数据集接口。
- `src/lerobot/datasets/dataset_metadata.py`：metadata 与统计信息。
- `src/lerobot/datasets/feature_utils.py`：feature shape/type 工具。
- `src/lerobot/datasets/streaming_dataset.py`：流式数据读取。
- `src/lerobot/datasets/video_utils.py`：视频后端。

#### ACT

- `src/lerobot/policies/act/configuration_act.py`：输入 feature、chunk、normalization 和模型超参数。
- `src/lerobot/policies/act/processor_act.py`：策略前后处理。
- `src/lerobot/policies/act/modeling_act.py`：`ACTPolicy`、`select_action`、`predict_action_chunk`、VAE、encoder/decoder 和 temporal ensembler。

ACT 默认对 action 使用 mean/std normalization。训练损失在 `ACTPolicy.forward` 中组合 action L1 与 VAE KL 项；必须结合 config 阅读，不能只看最终 loss。

#### Diffusion Policy

- `src/lerobot/policies/diffusion/configuration_diffusion.py`
- `src/lerobot/policies/diffusion/processor_diffusion.py`
- `src/lerobot/policies/diffusion/modeling_diffusion.py`

#### Training and evaluation

- `src/lerobot/scripts/lerobot_train.py`
- `src/lerobot/scripts/lerobot_eval.py`
- `src/lerobot/configs/train.py`
- `src/lerobot/utils/train_utils.py`

#### LIBERO wrapper

- `src/lerobot/envs/configs.py`：`LiberoEnv` 配置，默认 20 Hz。
- `src/lerobot/envs/libero.py`：suite/task 构建、相机映射、state/action space、reset 和 step。
- `src/lerobot/envs/factory.py`：统一环境工厂。

### LIBERO 数据契约（LeRobot 当前源码）

- 两个 RGB 相机：agent view 和 eye-in-hand。
- `envs.configs.LiberoEnv` 配置默认声明并向 factory 传入 360×360；底层 wrapper 构造函数的裸默认值
  是 256×256。通过 LeRobot config 创建时实际采用前者，smoke test 必须显式固定尺寸并核对 runtime。
- state dimension：8。
- action dimension：7。
- control frequency：20 Hz。
- 支持 relative 和 absolute control mode，必须与 checkpoint 一致。

### Phase 0-D6 resolver 实测

2026-08-25 在现有 `evla-core` 中对 pinned LeRobot 执行 pip dry-run：

```bash
python -m pip install --dry-run --no-cache-dir --no-build-isolation -e ".[libero]"
```

resolver 在 Python 3.12.14 上 PASS，并计划 59 项。已有 torch、torchvision、NumPy、SciPy、
Gymnasium 和 W&B 均保持原版本。主要新解析结果为：

| Package | Live pip result |
|---|---|
| `hf-libero` | 0.1.4 |
| `mujoco` | 3.8.1 CPython 3.12 Linux wheel |
| `robosuite` | 1.4.0 |
| `robomimic` | 0.2.0 |
| `transformers` | 5.5.4，满足 LeRobot `>=5.4,<5.6` |
| `opencv-python` | 5.0.0.93，阻断直接安装 |
| `tensorboard` | 2.21.0，由 robomimic 传递引入 |

该结果还说明 `uv.lock` 不是当前 pip 命令的强制 lock；实时 index 已使 Hydra、Numba、OpenCV、
TensorBoard 等版本发生漂移。后续安装要显式保护 core 包，并避免把 `opencv-python` 与当前
`opencv-python-headless==4.13.0.92` 混装。

`hf-libero==0.1.4` 和 `robosuite==1.4.0` 都声明 `opencv-python`，所以标准 resolver 无法表达
“由 headless wheel 提供同一 `cv2` runtime”。当前决策是：不直接安装 `.[libero]`；下一 gate
先形成手工 curated 依赖集合，预期对相关包使用受控 `--no-deps`，并把新增 metadata 例外与
运行时验证分开记录。

`hf-libero==0.1.4` 首次运行可能从 Hugging Face Hub 自动下载仿真 assets。LIBERO reset/render
不是无条件离线操作：在运行前先检查 cache，缺失时需要单独的数据量评估和下载授权。

宿主 preflight 已发现完整的 GLVND/NVIDIA EGL 静态链路，包括 EGL/OpenGL/GLX loader、
`libnvidia-eglcore.so.580.173.02`、`10_nvidia.json` 和 `/dev/nvidia*`。因此当前不需要先假设缺少
系统 EGL package。安装 Python runtime 后的最小 EGL context smoke 已实测 PASS：MuJoCo 3.8.1
从内联 XML 完成 10 个 physics steps，并经 NVIDIA device 0 返回 `(120,160,3)` `uint8` 的非恒定
finite RGB。该结果只覆盖独立 MuJoCo/EGL 链路，不覆盖 robosuite 或 LIBERO。

已检查的 LIBERO asset cache 均不存在。package 安装与 asset 下载必须拆成两个 gate；在仅安装
runtime 后可以先做 import/version 检查，但不能把首次 simulator reset 当作无下载命令。

安装后 wheel 源码静态检查进一步定位了 side effect：顶层 `import libero` 为空操作；
`import libero.libero` 只会在 `LIBERO_CONFIG_PATH` 缺失配置时创建目录并交互生成 YAML。
`get_assets_path()` 才是 `lerobot/libero-assets` 的自动下载入口，它由 LIBERO environment/object
实例化调用，而不是普通模块 import 调用。robosuite 的独立下载 helper 仅在对应脚本作为
`__main__` 时执行；当前 robosuite wheel 已随包包含约 586 MiB `models/assets`。无下载 import gate
据此使用临时配置目录与 Hugging Face offline mode，并禁止实例化 simulator。

### Curated installation boundary

两阶段 curated dry-run 已验证：

- stage 1 显式固定 torch、torchvision、NumPy、SciPy、Pillow、headless OpenCV、Gymnasium、
  W&B、Hugging Face Hub 等现有 core，并解析 55 个缺失 runtime distributions；
- stage 2 使用 `--no-deps`，只计划 `robosuite==1.4.0` 与 `hf-libero==0.1.4`；
- 两段均没有 `opencv-python`，没有 core upgrade/downgrade。

因此标准 `pip install -e ".[libero]"` 继续禁止，候选安装方式固定为“正常解析 curated dependencies，
以及两个顶层包使用 `--no-deps`”。这会有意牺牲三项 OpenCV metadata 完整性来保持唯一的 headless
`cv2` provider；必须把 package metadata gate 与 runtime gate 分开。

安装授权范围只包括 Python distributions，粗估压缩下载约 0.33 GiB，不包括 LIBERO assets。
安装前后记录 package manifest、环境目录大小和磁盘余量；若 core version 发生意外变化，停止在
import 前，不进入 simulator gate。

首次 Stage 1 暴露了一个 resolver dry-run 无法发现的 build-time constraint：
`egl_probe==1.0.2` 的 `CMakeLists.txt` policy minimum 早于 3.5，而 CMake 4.0 起移除了这段兼容
模式。当前环境的 CMake 4.1.3 因此在 configure 阶段失败。该失败发生在 pip wheel build 阶段，
其他已构建 wheel 没有安装，环境事务保持干净。

不采用降级 CMake 或 patch package source。CMake 4.0+ 官方提供的
`CMAKE_POLICY_VERSION_MINIMUM=3.5` 进程环境变量可以为新 build tree 初始化外部 policy floor；
已在当前 CMake 上用最小旧版 `cmake_minimum_required` 项目验证 default FAIL / variable PASS。
重试时变量只作用于单条 pip 命令。

Stage 1 重试已实际验证该修复：两个 EGL probe wheels 和其他 sdist 均成功构建，pip 安装 55 个
distributions。安装后 MuJoCo 3.8.1、robomimic 0.2.0、Transformers 5.5.4、Numba 0.67.0、
llvmlite 0.49.0 与 TensorBoard 2.21.0 可由 package metadata 查询。core pins 未漂移，
`opencv-python` 未安装，`pip check` 仍只有原 PushT 例外。

环境目录实测由 8.4 GiB 增至 9.0 GiB，Stage 1 落盘增量约 0.6 GiB。由于 `--no-cache-dir`，
下载 wheel 不保留到 pip cache；该数字不包含 Stage 2 的 robosuite 大型 wheel，也不包含 LIBERO
assets。

Stage 2 已按 `--no-deps` 边界成功安装 robosuite 1.4.0 和 `hf-libero` 0.1.4。实际 wheel 下载为
193.5 MB 与 3.2 MB，环境再增加约 0.6 GiB 至 9.6 GiB。两阶段总落盘增量约 1.2 GiB。
`opencv-python` 仍不存在，core pins 未变化。

当前 `pip check` 精确产生三条预期的 `opencv-python` 缺失报告，对应 PushT、robosuite、
`hf-libero`，无其他 dependency error。它证明 curated metadata 状态符合设计，但不能单独证明
共享的 headless `cv2` runtime 足以支持两个 simulator；接受例外仍取决于运行时回归。

Stage 2 后已完成第一部分运行时回归：`opencv-python` distribution 仍不存在，headless wheel 的
`cv2==4.13.0` 可正常 import 且 `GUI: NONE`；PushT wrapper、离屏 render、CUDA/BF16 与 ACT interface
全部再次 PASS。这排除了两个 OpenCV wheels 互相覆盖以及 core regression，但尚未证明 robosuite
和 LIBERO 的 image path 可由同一个 headless provider 满足。

随后在 LIBERO assets 缺失状态完成独立 MuJoCo EGL runtime gate：`MUJOCO_GL=egl`、device 0 下
simulation time 从 0.0 s 前进到 0.1 s，renderer 输出 `(120,160,3)`、`uint8`、range `0..255`、
finite 且非恒定图像。四个 LIBERO asset 候选路径后检仍不存在，因此没有发生隐式资产下载。

无下载 import boundary 也已实际 PASS：临时 `LIBERO_CONFIG_PATH`、`HF_HUB_OFFLINE=1` 下，
robosuite 1.4.0、`hf-libero` 0.1.4、LIBERO `OffScreenRenderEnv` 与 LeRobot `LiberoEnv` 均成功
导入；`libero_spatial` 暴露 10 tasks，task 0 的 BDDL 和 init-state 本地文件存在。唯一 OpenCV
runtime 仍是 headless 4.13.0，`GUI: NONE`。四个用户目录前后均不存在。robosuite 缺少
`macros_private.py` 的提示仅表示未设置用户覆盖宏，默认宏 import 正常，归类为预期警告且不执行
setup 脚本。由于未实例化 simulator，该结果仍不能代替 LIBERO EGL image-path 验证。

官方 Hugging Face `lerobot/libero-assets` 页面在 2026-08-25 显示 main 总体积 422 MB、verified
head `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`。仓库只包含 articulated/scanned/HOPE/TurboSquid
objects、scenes、textures 与少量顶层几何描述文件，不是 LIBERO demonstrations、训练数据或模型。
候选下载固定该 revision 并写入 `~/.cache/libero/assets`，按约 0.5 GiB 实际内容、1 GiB 安全预算
规划。当前约 114 GiB 可用，不构成容量阻断；资产下载继续作为单独授权 gate。

单独授权后的首次 `snapshot_download` 没有进入网络请求：当前 httpx 在 client factory 解析宿主
`socks://127.0.0.1:7897/` 时抛出 unknown proxy scheme。失败后目标 asset 目录、两个可能的 Hub
repo cache 与 `~/.libero` 均不存在，磁盘粗粒度无变化；这是 proxy preflight FAIL 且环境干净，
不是远端 repo/revision 或认证失败。下一步只读、脱敏检查大小写 proxy 变量和 SOCKS support，避免
打印潜在 credentials，也不安装包。

代理诊断结果：大小写 HTTP/HTTPS proxy 都使用 `http://127.0.0.1:7897`，大小写 ALL proxy 都使用
`socks://127.0.0.1:7897`，无 credentials。httpx 0.28.1 对 `http://` client construction PASS，
对 `socks://` 报 unknown scheme；改写成 `socks5://` 仍因 `socksio` 未安装而 FAIL，PySocks 也未
安装。由此选择零安装方案：仅对 Hugging Face 子进程取消 `ALL_PROXY`/`all_proxy`，让 HTTPS 使用
现有 HTTP proxy。先做 pinned repo metadata request，再重试 snapshot，避免直接重复大下载。

该 metadata request 已实际 PASS：repo ID 为 `lerobot/libero-assets`，resolved SHA 精确匹配固定
revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`，API 返回 586 sibling entries。四个候选用户路径
前后都不存在，证明没有 snapshot side effect。代理 workaround 与远端 revision 两项阻断已排除，
可在相同命令级环境下执行已授权下载。

固定 snapshot 随后下载 PASS：payload 586 files、422,320,936 bytes（402.76 MiB），HF local-dir
metadata 1,175 files / 179,955 bytes，目标总 `du` 409 MiB。六个预期 asset directories 均存在；
两个额外 Hub repo cache 与 `~/.libero` 均未创建。公共 repo unauthenticated warning 只说明 rate
limit，完整文件数、逻辑字节数与目录 contract 均通过。接下来的 simulator reset/render 使用
`HF_HUB_OFFLINE=1`，使资产 gate 从网络下载阶段切换为严格本地消费阶段。

严格离线 simulator gate 最终 PASS：LeRobot `LiberoEnv` 对 `libero_spatial` task 0 完成 deferred
construction、hard reset、固定 init state、10 次 settle、一次 7D dummy step、双相机 128×128
observation 和 `render()`。两路图像及 render 都是非恒定 finite `uint8` RGB，完整 robot state finite，
reset/step observation 与 action 均满足 Gym spaces。close 和临时 config cleanup PASS；asset manifest
前后保持 586 files / 422,320,936 bytes，`~/.libero` 未创建。该证据证明 headless `cv2` 足以支持
robosuite/LIBERO image path；三条 `opencv-python` 缺失 metadata 报告正式接受为预期例外。

Phase 0-D6 至此完成：标准 `.[libero]` resolver 仍因双 OpenCV provider 禁止；curated runtime、
Python 3.12 imports、独立 MuJoCo EGL、固定 assets 与 LeRobot LIBERO offline simulator 闭环均已验证。

最终状态审计也已 PASS：三个 submodule commits 与 handoff pins 精确一致，core/runtime package
versions 无漂移，唯一 OpenCV provider 仍为 headless 4.13.0.92，asset contract 精确保持 586 files /
422,320,936 bytes。环境 9.6 GiB、assets 409 MiB、`/home` 113 GiB free；`pip check` 只有三条正式
接受的 metadata 例外。Git 仅有三份项目文档修改且 `git diff --check` 无输出。

### 日志结论

当前源码原生集成 W&B，但全仓库未发现 `torch.utils.tensorboard.SummaryWriter` 或 TensorBoard 写入
路径。Phase 0 已验证 TensorBoard 2.21.0 primitive：event write/flush/close、两个 scalar tags 的
`EventAccumulator` 回读，以及同一步数 JSONL 双写全部 PASS；float32 scalar 尾差在 `allclose`
范围内。临时目录清理成功，没有启动 server 或 GUI。

最终决策：

- 本地运行产物放在 `.gitignore` 已覆盖的 `outputs/<experiment_id>/`；
- TensorBoard events 放 `tensorboard/`，同步的机器可读指标放 `metrics.jsonl`；
- resolved config、environment/package manifest 与 checkpoint manifest 使用同一 experiment ID；
- 可追踪的实验结论写入 Markdown experiment record，不提交 event files；
- Phase 1 首次训练前再实现或选择最小 metric sink，不在 Phase 0 修改上游 trainer；
- W&B login/真实 run 是外部服务动作，延后 Phase 1 并单独授权。

## 3. Original LIBERO

### 上游依赖事实

原始 README 使用：

- Python 3.8.13。
- PyTorch 1.11.0 + CUDA 11.3。
- NumPy 1.22.4。
- Transformers 4.21.1。
- robosuite 1.4.0、gym 0.25.2。

这套依赖与 Blackwell、LeRobot Python 3.12、NumPy 2 和 PyTorch cu128 不兼容。

### 决策

- 不执行 `libero/requirements.txt`。
- 不在 `evla-core` 中 editable-install 原始 LIBERO。
- 原始 submodule 用于阅读 benchmark、BDDL、环境、数据和评测语义。
- 实际 Phase 2 runtime 使用 LeRobot `libero` extra 提供的 `hf-libero`。
- smoke test 时核对 `hf-libero` 的 task、action、camera 和 success semantics 与原始仓库一致。

### 关键代码入口

- `libero/libero/benchmark/`：suite、task map 和固定初始化状态。
- `libero/libero/envs/bddl_base_domain.py`：任务、机器人、传感器与 20 Hz 控制环境。
- `libero/libero/envs/env_wrapper.py`：`OffScreenRenderEnv`。
- `libero/lifelong/datasets.py`：原始 HDF5/robomimic sequence dataset wrapper。
- `libero/lifelong/main.py`：训练入口。
- `libero/lifelong/metric.py`：success evaluation。
- `libero/lifelong/evaluate.py`：独立评测入口。

原始最小示例使用 7 维 dummy action 和 sparse success reward。这将作为 LeRobot wrapper smoke test 的语义对照。

## 4. OpenVLA

### 上游依赖事实

OpenVLA `0.0.3` 的 `pyproject.toml` 固定：

| Component | Pin |
|---|---|
| Python | `>=3.8`，官方开发环境 3.10 |
| PyTorch | `==2.2.0` |
| torchvision | `==0.17.0` |
| torchaudio | `==2.2.0` |
| Transformers | `==4.40.1` |
| tokenizers | `==0.19.1` |
| timm | `==0.9.10` |
| PEFT | `==0.11.1` |
| FlashAttention | README 指定 `2.5.5` |

完整 editable install 还会安装 TensorFlow、TFDS、tensorflow-graphics 和 Git 依赖 `dlimp`。这些主要服务 RLDS/训练，并非最小推理都需要。

### 与本机的冲突

- PyTorch 2.2 官方组合早于 Blackwell 支持，不能作为 RTX 5070 Ti runtime。
- OpenVLA Transformers 4.40 与 LeRobot Transformers 5.4–5.5 互斥。
- FlashAttention 2.5.5 不是本机第一轮应编译的目标。
- 因此 OpenVLA 必须使用独立 `evla-openvla` 环境。

### Phase 3 安装决策

- 不直接执行 `python -m pip install -e openvla`，否则会尝试固定 PyTorch 2.2。
- 先创建 curated minimal-inference requirements。
- 显式安装支持 Blackwell 的 cu128 PyTorch。
- 保留 Transformers/tokenizers/timm 的 OpenVLA 兼容版本，逐项 smoke test。
- local package 如需 editable install，使用 `--no-deps`，依赖由我们显式控制。
- attention 首先使用 eager/SDPA；不安装 FlashAttention 2.5.5。

### 显存判断

OpenVLA 自带 verification script 注释给出：

- 8-bit：约 9 GB passive / 10 GB active。
- 4-bit：约 6 GB passive / 7 GB active。

对 12 GB Laptop GPU，4-bit 是首选；8-bit 的剩余空间过小。该数字只是上游脚本说明，必须在本机实测峰值显存。

官方普通 LoRA 建议至少约 27 GB VRAM，不适合本机。`vla-scripts/finetune.py` 支持 `use_quantization=True` 的 4-bit LoRA，但 12 GB 上能否完成 backward 尚未验证，因此归类为 go/no-go 实验。

### 关键代码入口

- `prismatic/`：VLM/VLA backbone、数据和训练基础设施。
- `prismatic/vla/action_tokenizer.py`：动作 tokenization。
- `prismatic/models/vlas/openvla.py`：动作预测和反归一化。
- `vla-scripts/extern/verify_openvla.py`：BF16/8-bit/4-bit 推理样例。
- `vla-scripts/finetune.py`：LoRA/4-bit LoRA。
- `experiments/robot/libero/run_libero_eval.py`：LIBERO rollout 与 4-bit flag。

## 5. 最终环境边界

```text
evla-core
  Python 3.12
  PyTorch 2.11 + cu128 candidate
  LeRobot 0.6.2
  Phase 1: training + pusht
  Phase 1B: + diffusion
  Phase 2: + libero (hf-libero)

evla-openvla
  Python 3.10
  Blackwell-compatible cu128 PyTorch
  OpenVLA-compatible Transformers 4.40 stack
  4-bit inference first
  no FlashAttention first
```

选择 PyTorch 2.11 作为 `evla-core` 候选，因为它是 LeRobot 当前 `<2.12` 范围内的最新受支持版本，并有 cu128 wheel。只有 resolver dry-run 和实际 CUDA smoke test 通过后才写入最终 lock。

## 6. 下一步

1. 单独授权后提交 Phase 0 文档记录。
2. 进入 Phase 1 前先形成 ACT baseline 数据与训练 gate；下载、训练、W&B 和上传分别授权。
