# ACT Baseline

状态：环境与模型接口已验证，数据和训练尚未开始。

## 目的

建立项目第一个 image/state → action chunk 模仿学习闭环。先在 PushT 上理解 LeRobot 数据、ACT、normalization 和 rollout，再迁移到一个 LIBERO task。

## 为什么先选 PushT

- 二维连续动作便于检查 action semantics。
- 图像、agent state 和 simulator state 可以分别观察。
- 无语言、无复杂机器人关节和无 MuJoCo 依赖，适合隔离 LeRobot dataset/policy pipeline。
- 96×96 图像和 51.59M ACT 适合 12 GB 单 GPU 做小规模训练。

## Phase 0 已验证接口

### Observation contract

| Key | Runtime shape | Dtype/range | Meaning |
|---|---|---|---|
| raw `pixels` | `[B,96,96,3]` | uint8, 0–255 | PushT top-view RGB |
| `observation.image` | `[B,3,96,96]` | float32, 0–1 | ACT visual input |
| raw `agent_pos` | `[B,2]` | float64 | agent x/y |
| `observation.state` | `[B,2]` | float32 | ACT state input |

### Action contract

| Item | Value |
|---|---|
| dimension | 2 |
| semantics | absolute target position `[x,y]` |
| environment range | `[0,512]` per dimension |
| environment control rate | 10 Hz |
| ACT `chunk_size` smoke config | 100 |
| ACT `n_action_steps` smoke config | 10 |
| training normalization | 待读取 dataset statistics 后确定 |

### Model interface result

| Item | Result |
|---|---|
| Backbone | ResNet18 |
| Parameters | 51.59 M |
| Random input batch | image `[1,3,96,96]` + state `[1,2]` |
| Random output | action chunk `[1,100,2]` |
| First-pass latency | 0.282970–0.309598 s，两次接口检查 |
| Peak PyTorch allocation | 222.56–223.43 MiB |
| Status | PASS |

上述延迟包含首次 warm-up，显存仅为 FP32 inference，随机动作没有控制意义。这些数值不进入训练结果或简历定量成果。

## 关键源码链路

```text
PushtEnv / make_env
  → preprocess_observation
  → ACTConfig input_features/output_features
  → ACTPolicy.predict_action_chunk
  → ACT model: ResNet18 + transformer encoder/decoder
  → [batch, chunk_size, action_dim]
```

需要重点理解：

- `lerobot/src/lerobot/envs/configs.py`：环境声明与 Gym kwargs。
- `lerobot/src/lerobot/envs/utils.py`：HWC uint8 到 BCHW float32。
- `lerobot/src/lerobot/policies/act/configuration_act.py`：feature、chunk 和网络超参数。
- `lerobot/src/lerobot/policies/act/modeling_act.py`：action queue、VAE latent 和 action chunk。

## 已知工程问题

Pinned LeRobot 的 PushT 默认 feature 声明为 384×384，但 `gym-pusht==0.1.6` runtime observation 为 96×96。基线固定使用 `configs/env/pusht_96.yaml`，不修改上游 submodule。

## 计划步骤

1. 经授权后只下载 PushT 当前实验所需数据。
2. 检查一个 episode/sample 的 key、shape、dtype、FPS 和 episode index。
3. 读取 dataset statistics，解释 state/action normalization 与反归一化。
4. 核对 ACT 的 action delta indices、padding mask 和 chunk sampling。
5. 建立极小 step 的 overfit/smoke 配置，但不自动训练。
6. 通过 gate 后再运行正式单 GPU训练。
7. 保存曲线、checkpoint manifest、rollout video 和失败案例。
8. 在相同评测协议下迁移到一个 LIBERO task。

## 第一轮训练前必须决定

- Dataset repo ID 与精确 revision。
- 训练/验证 episode split。
- `chunk_size` 是否使用 100，或根据 10 Hz task horizon 缩短。
- `n_action_steps` 与 rollout replanning frequency。
- pretrained ResNet18 是否允许下载。
- batch size、BF16、gradient accumulation 和峰值显存 gate。
- W&B run name 与本地 TensorBoard/JSON logging 方案。

## 必须输出

- 数据检查报告。
- 训练配置与完整命令。
- loss curve。
- success metric 与评测 episode 数。
- warm-up 后 inference latency/FPS。
- peak training VRAM 和 wall-clock time。
- checkpoint manifest。
- 至少一个成功和一个失败 rollout。
- 对 observation、normalization 和 action chunk 的代码级解释。
