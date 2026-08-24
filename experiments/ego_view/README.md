# Ego-view Extension

状态：planned。

## 目的

研究从第三视角到 wrist/ego camera 时，视角、遮挡、坐标和图像分布变化如何影响机器人操作策略。

## 受控变量

- camera pose 与 field of view。
- third-person / ego / multi-view。
- 图像裁剪、颜色、模糊和几何增强。
- action frame 与 camera frame 的关系。
- observation latency 和运动模糊。

## 实验原则

固定任务、数据量、策略、seed 和训练预算，只改变 camera setup；禁止把多项改动混在同一组对照中。

## 必须输出

- camera 配置和坐标图。
- viewpoint ablation table。
- 遮挡与视角失败案例。
- 对 Feagine 相机布置的设计建议。
