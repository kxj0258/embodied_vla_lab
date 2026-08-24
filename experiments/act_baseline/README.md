# ACT Baseline

状态：planned。

## 目的

建立项目第一个 image/state → action chunk 模仿学习闭环。先在 PushT 上理解 LeRobot 数据、ACT 和 rollout，再迁移到单个 LIBERO 任务。

## 学习重点

- observation image/state feature。
- action chunk、padding 和 temporal aggregation。
- normalization 与反归一化。
- train/eval 数据一致性。
- checkpoint、W&B/TensorBoard、视频和成功指标。

## 计划步骤

1. 检查一个 PushT sample 的 key、shape、dtype 和时间窗口。
2. 阅读 ACT config 与核心模型调用链。
3. 建立小步数 overfit/smoke test。
4. 通过后再运行正式单 GPU 训练。
5. 保存曲线、checkpoint manifest 和 rollout video。
6. 在相同评测协议下迁移到一个 LIBERO-Spatial 任务。

## 必须输出

- 训练配置与完整命令。
- loss curve。
- success metric。
- inference latency/FPS。
- peak VRAM 和 wall-clock time。
- 至少一个成功和一个失败 rollout。
- 对 action chunk 的代码级解释。
