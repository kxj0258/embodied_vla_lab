# System Architecture

## 1. 系统边界

Embodied VLA Lab 将环境、数据、策略和机器人控制解耦，避免将某个仓库的内部表示直接扩散到整个项目。

```text
┌─────────────────────────────────────────────────────────────┐
│ LIBERO / PushT / Feagine MuJoCo environments                │
│ raw observation, raw action, task, reward, done, success    │
└──────────────────────────┬──────────────────────────────────┘
                           │ environment adapter
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Canonical data/action contract                              │
│ images, state, language, action, timestamp, metadata        │
└───────────────┬──────────────────────────┬──────────────────┘
                │                          │
                ▼                          ▼
┌───────────────────────────┐  ┌──────────────────────────────┐
│ LeRobotDataset v3         │  │ RLDS / OpenVLA data loader  │
│ ACT / Diffusion Policy    │  │ VLM + action tokenizer      │
└───────────────┬───────────┘  └──────────────┬───────────────┘
                │ policy action               │ policy action
                └──────────────┬───────────────┘
                               ▼
                   action postprocessor
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
       LIBERO controller             Feagine PCC controller
```

## 2. 三个上游仓库

### LeRobot

工程主干，负责统一 dataset、policy、processor、training、checkpoint 和 evaluation。实验优先调用公开接口；若需要策略或环境扩展，优先采用 out-of-tree adapter/plugin，避免维护大规模 fork。

### LIBERO

提供 MuJoCo 场景、语言任务、示范、任务套件和成功判定。原始仓库用于理解 benchmark 语义；实际 Blackwell 运行环境优先使用 LeRobot 当前维护的兼容包，并核对它没有改变任务定义和评测协议。

### OpenVLA

提供图像与语言到动作的生成模型。其输出动作依赖数据集 normalization statistics，因此 checkpoint、`unnorm_key`、action dimension 和控制语义必须作为一组版本化。

## 3. Canonical observation contract

建议统一表示如下，具体 shape 在读取真实数据后冻结：

```yaml
observation:
  images:
    scene: uint8[H, W, 3]
    wrist: optional uint8[H, W, 3]
  state: float32[state_dim]
  language: optional string
  timestamp: float64
action:
  value: float32[action_dim]
  representation: required string
  frame: required string
  frequency_hz: required number
metadata:
  episode_index: int
  task_id: string
  success: optional bool
```

禁止仅记录一个无语义的 `action: float[7]`。必须同时记录：

- absolute 或 delta。
- joint、end-effector 或 actuator space。
- translation/rotation frame。
- rotation representation。
- gripper convention。
- control frequency 和 action horizon。
- normalization statistics 及其来源。

## 4. Policy contract

所有策略至少支持以下逻辑接口：

```text
preprocess(raw observation)
        ↓
predict_action or predict_action_chunk
        ↓
postprocess(normalized policy output)
        ↓
environment/robot action
```

ACT 和 Diffusion Policy 输出 action chunk；OpenVLA 基线通常按动作 token 逐步生成。评测层应隐藏这种差异，并统一记录：

- observation timestamp。
- inference latency。
- returned action horizon。
- actually executed action count。
- control-loop FPS。

## 5. Feagine transfer boundary

Feagine 是连续体机械臂，不能默认复用刚性机械臂的 joint action。第一版桥接采用高层末端任务空间：

```text
VLA output
  → desired tip pose / delta tip pose
  → workspace and collision constraint
  → PCC inverse kinematics
  → curvature/section variables
  → actuator controller
```

如果 VLA 输出 LIBERO 风格 7D delta end-effector action，adapter 负责坐标变换、限幅、时间重采样和 gripper/interaction 语义映射。PCC IK 和低层控制保持可独立测试。

## 6. Repository ownership

| 路径 | 职责 | 是否允许大规模生成文件 |
|---|---|---|
| `lerobot/` | 固定上游源码 | 否 |
| `libero/` | 固定上游源码 | 否 |
| `openvla/` | 固定上游源码 | 否 |
| `experiments/` | 实验入口与分析 | 仅小文件 |
| `configs/` | 可复现配置 | 否 |
| `scripts/` | 数据检查、评测和工具脚本 | 否 |
| `checkpoints/` | 本地权重 | 是，不进入 Git |
| `results/` | 指标摘要与报告 | 原始大文件不进入 Git |

## 7. Design invariants

- 环境 action 和 policy action 之间必须存在显式 adapter。
- 所有 normalization 均需版本化并可反变换。
- 每个 rollout 必须记录 seed、task、checkpoint 和 commit。
- 模型效果与系统延迟分开评估。
- 仿真通过前不接入真实机器人。
