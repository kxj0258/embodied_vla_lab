# Feagine Transfer

状态：Phase 6A通用静态bridge PASS；Phase 6B A03 Type-2资产合同PASS；Phase 6C隔离zero-step runtime PASS；Phase 6D短时动力学诊断PASS；Phase 6J纠正目标严格2/4；Phase 6L安全裕量扫描未选出通过0.1 mm门的command。

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

## Phase 6A边界

当前实现是单段可伸缩PCC参考模型，用于验证接口、坐标、限幅和失败安全，不是Feagine A03数字孪生。
实际A03 MJCF、section/actuator定义、标定几何和控制周期未在工作区中找到。在这些资产绑定并通过独立静态门之前，不运行MuJoCo，也不声称机器人或仿真有效性。

后续已定位`feagine-simulation/dist`发行包并完成静态绑定：A03 Type-2为3个固定0.16m PCC section，core
IK输入为位置+tip direction，输出6个section θ/φ；MJCF timestep为0.005s。wheel默认preset `a03`与实际
唯一目录`a03_type_2`不一致，项目调用必须显式指定后者。隔离环境已完成core/wheel安装、MJCF绑定与
zero-step验证。200-step动力学v1的严格joint-limit检查保留FAIL；相同协议v2确认仅夹爪slide关节软限位
穿透11.77 μm，低于预注册0.1 mm材料阈值。后续纠正可达性与signed-θ表示后四方向均产生目标运动，
但严格安全结果为2/4；夹爪margin扫描将±X瞬态由约0.2631 mm降至0.1150 mm，仍未通过0.1 mm门。
这些诊断不等价于VLA策略迁移或硬件成功。
