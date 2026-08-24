# Compatibility Matrix

更新时间：2026-08-25。

本文件区分“宿主机事实”“已通过运行时验证”和“后续计划”。

## 宿主机

| 项目 | 结果 | 状态 |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS | 已验证 |
| Kernel | 6.8.0-138-generic | 已验证 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 已验证 |
| VRAM | 12227 MiB | 已验证 |
| GPU architecture | Blackwell, compute capability 12.0 | 已验证 |
| NVIDIA driver | 580.173.02 | 已验证 |
| Driver CUDA ceiling | 13.0 | 已验证，不等于 Toolkit |
| RAM | 32 GB（用户提供） | 待命令复核 |
| Free disk at initial audit | 约 126 GB | 已验证 |
| Desktop session | X11 | 已验证 |
| conda | Miniforge, conda 26.5.3 | 已验证 |
| nvcc / system CUDA Toolkit | 未安装 | 当前不阻塞 |

Codex 沙箱不可访问宿主 GPU，CUDA 命令由用户在宿主终端执行。GPU 工作前重新检查其他进程占用，不把关闭 Unitree 程序作为永久环境要求。

## 关键解释

`nvidia-smi` 的 CUDA 13.0 是驱动可支持的最高 CUDA runtime，不表示系统安装了 CUDA 13 Toolkit。当前预编译 PyTorch wheel 自带 CUDA 12.8 runtime；在必须编译 CUDA extension 前不安装系统 Toolkit。

## 固定上游版本

| Repository | Commit |
|---|---|
| LeRobot | `8b256a6c0d4769c3cc3e7e98f04940126398a391` |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| OpenVLA | `c8f03f48af692657d3060c19588038c7220e9af9` |

## Environment A：evla-core

用途：LeRobot、PushT、ACT、Diffusion Policy、LIBERO、MuJoCo、ego-view 和小型 world model。

| Component | Locked/verified value | Status |
|---|---|---|
| Python | 3.12.14 | 已验证 |
| pip | 26.2.1 | 已验证 |
| FFmpeg | 7.1.1, conda-forge | 已验证 |
| PyTorch | 2.11.0+cu128 | CUDA/BF16 已验证 |
| torchvision | 0.26.0+cu128 | ACT forward 已验证 |
| CUDA architecture | `(12,0)`, `sm_120` | 已验证 |
| LeRobot | 0.6.2 editable pinned source | 已验证 |
| gym-pusht | 0.1.6 | headless reset/step/render 已验证 |
| OpenCV | headless 4.13.0.92 only | 已验证 |
| W&B | 0.27.2 | import 已验证；login/run 延后 Phase 1 并单独授权 |
| TensorBoard | 2.21.0，由 robomimic 传递安装 | event write/read + JSONL roundtrip 已验证 |
| MuJoCo EGL | 3.8.1；NVIDIA EGL physics step 与离屏 RGB render PASS | 已验证 |
| LIBERO runtime | `hf-libero==0.1.4`、robosuite 1.4.0；offline reset/step/双相机 EGL RGB PASS | 已验证 |

原始 LIBERO 的 Python 3.8、PyTorch 1.11、CUDA 11.3 requirements 不进入该环境。实际 runtime 计划使用 LeRobot 当前 `libero` extra 的 `hf-libero` compatibility package，安装前重新 dry-run。

### Phase 0-D6 resolver 结论

2026-08-25 宿主 pip dry-run 对 `-e ".[libero]"` 完整解析成功，计划项共 59 个。核心版本
`torch==2.11.0+cu128`、`torchvision==0.26.0+cu128`、`numpy==2.2.6`、
`scipy==1.18.1`、`gymnasium==1.3.0` 和 `wandb==0.27.2` 均保持不变。

关键候选为：

- `hf-libero==0.1.4`；
- `mujoco==3.8.1`，有 CPython 3.12 Linux wheel；
- `robosuite==1.4.0`、`robomimic==0.2.0`；
- `transformers==5.5.4`；
- `opencv-python==5.0.0.93`。

resolver PASS 只证明 metadata 可解，不证明 Python 3.12 runtime、EGL 或 simulator 正常。直接安装计划
因最后一项 OpenCV 冲突而判为 blocked；必须改用 curated 依赖集合并再次 dry-run。

锁文件参考闭包的已知压缩构件约 402 MiB，实际实时 resolver 版本已有漂移，最终下载量和落盘增量
需在 curated dry-run 后重估。`hf-libero` 首次运行还可能自动下载 LIBERO assets；assets 不属于
Python package 安装授权，需单独检查缓存和确认。

宿主静态 preflight 已确认：`libEGL.so.1`、`libOpenGL.so.0`、`libGLX.so.0`、
`libnvidia-eglcore.so.580.173.02`、NVIDIA GLVND vendor JSON 和 `/dev/nvidia*` 均可见。
`MUJOCO_GL` 等变量当前未设置，后续 smoke 使用命令级临时环境变量。该结论不替代真实 EGL context
创建与 RGB render。

LIBERO asset cache 在 `~/.cache/libero/assets`、两个可能的 Hugging Face Hub repo cache 和
`~/.libero` 中均未发现。该 preflight 当时 `/home` 可用 115 GiB，随后 assets 已经单独授权并按固定
revision 下载；最终状态见下文。

### 已知例外：OpenCV package metadata

LeRobot core 使用 `opencv-python-headless`，`gym-pusht==0.1.6` metadata 则要求 `opencv-python`。为了避免同一个 `cv2` namespace 的两个 wheel 共存，PushT 使用 `--no-deps` 安装并保留 headless wheel。

Phase 0-D6 curated 安装后，robosuite 与 `hf-libero` metadata 也声明非 headless wheel。当前
`pip check` 有且仅有下面三条已接受例外：

```text
gym-pusht 0.1.6 requires opencv-python, which is not installed.
robosuite 1.4.0 requires opencv-python, which is not installed.
hf-libero 0.1.4 requires opencv-python, which is not installed.
```

如果出现其他 `pip check` 错误，不能归入该例外。

禁止安装 resolver 选出的 `opencv-python==5.0.0.93`。当前唯一 `cv2==4.13.0` 来自
`opencv-python-headless==4.13.0.92`，且 `GUI: NONE`；PushT reset/step/render、Phase 0 core 回归、
robosuite/LIBERO import 以及 LIBERO 双相机 EGL reset/step/render 均已 PASS。三条 metadata 报告
因此有完整运行时证据，正式作为长期预期例外接受。

两阶段 curated dry-run 已通过：第一阶段计划 55 个 dependencies，第二阶段用 `--no-deps` 严格
计划 `robosuite==1.4.0` 与 `hf-libero==0.1.4`，两段均未出现 `opencv-python`，且 core 版本无
变化。该方案在实际安装前仍需单独授权。

首次 Stage 1 实际安装在 `egl_probe==1.0.2` wheel build 阶段被 CMake 4.1.3 拒绝，因为上游
声明的 policy version 低于 3.5。pip 未进入安装事务，环境保持原状。已用当前 CMake 最小复现并
验证进程级 `CMAKE_POLICY_VERSION_MINIMUM=3.5` 可以通过配置；后续重试采用该变量，不降级
CMake，也不修改第三方源码。

使用该变量重试后 Stage 1 PASS，成功安装 55 个 distributions。MuJoCo 3.8.1、robomimic 0.2.0、
Transformers 5.5.4 和 TensorBoard 2.21.0 已落盘；core pins 与唯一 headless OpenCV provider 均
保持不变。环境从 8.4 GiB 增至 9.0 GiB。该结果是 package install PASS，不是 import、EGL、
TensorBoard logging 或 LIBERO simulator PASS。

实际安装后预期 `pip check` 的 OpenCV metadata 例外将从一条变为三条：

```text
gym-pusht 0.1.6 requires opencv-python, which is not installed.
hf-libero 0.1.4 requires opencv-python, which is not installed.
robosuite 1.4.0 requires opencv-python, which is not installed.
```

Stage 2 后及 LIBERO simulator gate 后均实测恰好出现上述三条，没有第四条错误。

Stage 2 实际下载 robosuite 193.5 MB 与 `hf-libero` 3.2 MB，环境由 9.0 GiB 增至 9.6 GiB。
完整 D6 package 安装使环境由 8.4 GiB 增至 9.6 GiB；在该 Stage 2 时间点 LIBERO assets 尚未下载。

Stage 2 后 headless OpenCV 与 core 回归已 PASS：非 headless distribution 不存在，`cv2==4.13.0`
且 `GUI: NONE`，PushT reset/step/render 与 ACT interface 四个固化 gate 全部通过。

独立 MuJoCo EGL smoke 也已 PASS：命令级设置 `MUJOCO_GL=egl` 和
`MUJOCO_EGL_DEVICE_ID=0` 后，最小 XML 从 0.0 s 前进至 0.1 s，并返回
`(120,160,3)`、`uint8`、range `0..255` 的非恒定 finite RGB。测试前后 LIBERO 四个候选 asset
目录仍均不存在，因此该结果验证 MuJoCo/NVIDIA EGL 链路，但不冒充 LIBERO simulator 验证。

LIBERO 无下载 import boundary 已 PASS：robosuite、`OffScreenRenderEnv`、LeRobot `LiberoEnv`、
`libero_spatial` 10-task metadata 均可加载，task 0 的 BDDL 与 init-state 文件存在，OpenCV 仍为
`GUI: NONE`。临时 `LIBERO_CONFIG_PATH` 和 Hugging Face offline mode 下四个用户路径前后均不存在。
robosuite 关于缺少 `macros_private.py` 的三行输出是可选用户覆盖宏提示，归为预期警告；不运行其
setup 脚本。该 gate 不是 simulator 实例化/render 验证。

官方 `lerobot/libero-assets` 在 2026-08-25 显示 422 MB，计划固定 verified revision
`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`，落盘目标 `~/.cache/libero/assets`。内容是 simulator
几何、纹理和 scene assets，不包含 demonstrations、训练 dataset 或模型。该段是下载前预算；后续
已获得单独授权并完成固定 snapshot。

首次已授权下载在 httpx 初始化时因宿主代理 `socks://127.0.0.1:7897/` 的未知 scheme 失败，尚未
发出 repo tree 请求。目标目录及其他三个用户路径均未创建，判为 clean FAIL；需先确认代理变量与
SOCKS transport capability，不把它归因于 Hugging Face repo 或 revision。

脱敏诊断已确认 HTTP/HTTPS proxy 均为可接受的 `http://127.0.0.1:7897`，冲突来自额外的大小写
`ALL_PROXY=socks://127.0.0.1:7897`。httpx 0.28.1 对 HTTP client init PASS；SOCKS scheme FAIL，
且 `socksio`/PySocks 均未安装。选择命令级取消两项 ALL proxy、保留 HTTP/HTTPS proxy，不安装
新包、不修改永久环境；网络 metadata gate 尚待执行。

命令级取消 ALL proxy 后的 pinned metadata gate 已 PASS：Hugging Face 返回同一 repo ID、完整 SHA
`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` 和 586 个文件条目；四个 asset/config 路径前后不变。
因此 HTTP proxy 连通性与 revision 均已确认，可以在相同边界下重试已授权 snapshot。

固定 snapshot 下载现已 PASS：586 payload files、422,320,936 bytes（402.76 MiB），本地 `du`
为 409 MiB；六个必需 asset directories 全部存在。没有创建额外 models/datasets Hub repo cache，
也没有创建 `~/.libero`；公共仓库 unauthenticated 提示仅影响限速。后续 LIBERO simulator 运行固定
为 Hugging Face offline mode，避免任何隐式网络访问。

最终 LIBERO simulator gate 已 PASS：`libero_spatial` task 0 在单环境、128×128 双相机、20 Hz、
relative control 下完成 reset、10 次 settle、一次 7D dummy step 和 LeRobot `render()`；reset/step
observations 均匹配 Gym spaces，robot state finite，双相机及 render 均为非恒定 `uint8` RGB。
environment close、临时 config 清理与 asset manifest 不变均通过，且 `~/.libero` 仍不存在。

D6 最终只读审计确认三项 submodule pins、core/runtime versions、唯一 headless OpenCV 和固定 asset
manifest 均无漂移；环境 9.6 GiB、assets 409 MiB、`/home` 113 GiB free。`pip check` 仍恰好三条
已接受例外，Git 仅修改本文件、phase 记录和 dependency analysis，`git diff --check` PASS。

本地日志 smoke 也已 PASS：`torch.utils.tensorboard.SummaryWriter` 生成一个 296-byte event file，
`EventAccumulator` 回读两个 tags 和 steps `[0,1]`；JSONL 两条记录完全 roundtrip。learning-rate
回读的微小尾差来自 event scalar 的 float32 表示，`allclose` 通过。测试只使用临时目录，完成后
删除，没有启动 server/GUI。后续本地布局固定为忽略版本控制的
`outputs/<experiment_id>/{tensorboard/,metrics.jsonl,...}`；trainer integration 延后 Phase 1。

### 已知例外：PushT feature declaration

Pinned LeRobot `PushtEnv` 默认声明 `(3,384,384)` image feature，但其 `gym_kwargs` 没有向 `gym-pusht==0.1.6` 传 observation width/height，实际图像为 `(3,96,96)`。项目基线使用 `configs/env/pusht_96.yaml` 显式对齐声明和 runtime，不修改上游 submodule。

## Environment B：evla-openvla

到 Phase 3 再创建。

| Component | Candidate | Status |
|---|---|---|
| Python | 3.10 | 计划 |
| PyTorch | Blackwell-compatible cu128 build | 待验证 |
| Quantization | 4-bit bitsandbytes | 计划 |
| Batch size | 1 | 计划 |
| Attention | eager/SDPA first | 计划 |
| FlashAttention | 第一轮不安装 | 已决定 |

Pinned OpenVLA 将 PyTorch 固定为 2.2.0、Transformers 固定为 4.40.1，并建议 FlashAttention 2.5.5。该组合不能原样用于 Blackwell，也不能与 LeRobot 的新依赖栈共存。Phase 3 使用独立 curated minimal environment，不直接执行普通 editable install。

## Resource budget

- 单 GPU，12 GB VRAM；训练优先 BF16、小 batch 和梯度累积。
- OpenVLA 优先 4-bit inference；QLoRA 先做显存 go/no-go。
- conda 环境、数据、checkpoint 和视频均需控制磁盘预算。
- 一次只保留当前阶段需要的数据子集和大型 checkpoint。
- 权重、数据和大视频不进入 Git。

## 待验证清单

- [ ] `free -h` 复核 RAM 与 swap。
- [x] 空闲状态可用显存约 10.7 GB（一次实测，任务前仍需复查）。
- [x] PyTorch cu128 识别 compute capability 12.0。
- [x] BF16 tensor 运算。
- [x] LeRobot import、PushT reset/step/render 和 ACT forward。
- [x] 固化的 Phase 0 core script 在宿主终端完整复核。
- [x] MuJoCo/LIBERO package 安装后再次完成 core、PushT、ACT 与 headless OpenCV 回归。
- [x] MuJoCo EGL renderer。
- [x] LIBERO reset/step/离屏图像。
- [x] LeRobot `libero` extra 在 Python 3.12 中完成 resolver dry-run。
- [x] 宿主 EGL loader、GLVND NVIDIA vendor 和 `/dev/nvidia*` 静态 preflight。
- [x] 排除双 OpenCV wheel 的两阶段 curated resolver dry-run。
- [x] 检查并经单独授权下载固定 revision 的 LIBERO simulator assets。
- [ ] W&B login、offline/online run。
- [x] LeRobot 当前源码未发现原生 TensorBoard SummaryWriter。
- [ ] 决定 TensorBoard callback 或离线转换方案。

## 上游参考

- LeRobot: <https://github.com/huggingface/lerobot>
- LIBERO: <https://github.com/Lifelong-Robot-Learning/LIBERO>
- OpenVLA: <https://github.com/openvla/openvla>
- PyTorch Blackwell support: <https://pytorch.org/blog/pytorch-2-7/>
- NVIDIA CUDA architecture matrix: <https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html>
