# Feagine Transfer

状态：planned；simulation first。

## 目的

将已验证的视觉/语言策略输出迁移为 Feagine A03 连续体机械臂可以安全执行的高层目标，并通过 PCC IK/controller 在 MuJoCo 中闭环验证。

## 建议接口

```text
image + instruction
        ↓
VLA / policy high-level action
        ↓
coordinate transform + scaling + safety limits
        ↓
desired tip pose / delta tip pose
        ↓
PCC inverse kinematics
        ↓
section/actuator command
        ↓
MuJoCo simulation
```

## 初始任务

1. Move to target。
2. Avoid obstacle。
3. Contact interaction。

## 接入前需要明确

- Feagine MJCF 入口和 actuator 定义。
- PCC state、forward/IK API。
- 末端位姿与坐标系。
- 控制周期、限幅和碰撞约束。
- camera 内外参和 observation latency。

## 必须输出

- action adapter 接口文档。
- 单元测试和动作范围验证。
- MuJoCo 闭环 rollout。
- 失败安全策略。
- 与刚性机械臂 action space 的差异分析。
