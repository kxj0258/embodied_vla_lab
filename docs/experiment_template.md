# Experiment Record Template

> 复制本模板到对应实验目录，文件名建议为 `YYYY-MM-DD_<experiment_id>.md`。任何未实际测量的结果必须标记为“待测”，不能填入预计数值。

## 1. 基本信息

- Experiment ID：
- 日期：
- 阶段：
- 负责人：kxj_jiaolong
- 状态：planned / running / completed / failed
- 对应 issue/任务：

## 2. 实验目的

用一两句话说明本实验要回答的工程或研究问题。

## 3. 假设与成功标准

### 假设

- H1：
- H2：

### 通过标准

- 功能 gate：
- 定量指标：
- 资源限制：

## 4. 环境

- OS / kernel：
- GPU / driver：
- Python：
- PyTorch / CUDA runtime：
- conda environment：
- LeRobot commit：
- LIBERO commit/package：
- OpenVLA commit/checkpoint：
- 其他关键依赖：

附环境快照或其路径：

```bash
# 建议记录命令；执行后再粘贴结果摘要
python --version
python -m pip freeze
```

## 5. 数据

- Dataset ID / version：
- 本地路径或 Hub ID：
- 使用 episode 数：
- train/validation/test split：
- 图像相机与分辨率：
- 数据 FPS：
- augmentation：
- 数据过滤：

### Observation contract

| Key | Shape | Dtype | Range/normalization | Meaning |
|---|---|---|---|---|
|  |  |  |  |  |

### Action contract

| Item | Value |
|---|---|
| dimension |  |
| absolute/delta |  |
| joint/EEF/actuator |  |
| frame |  |
| rotation representation |  |
| gripper convention |  |
| frequency |  |
| chunk horizon |  |
| normalization |  |

## 6. 方法与配置

- Policy：
- Backbone：
- Trainable parameters：
- Precision：
- Batch size：
- Gradient accumulation：
- Optimizer / LR：
- Training steps：
- Seed：
- Evaluation episodes：

配置文件：

```text
configs/...
```

## 7. 命令

### 数据检查

```bash
# command
```

### 训练

```bash
# command，只有得到明确授权后才执行
```

### 评测

```bash
# command
```

## 8. 结果

| Metric | Value | Protocol/notes |
|---|---:|---|
| training loss | 待测 |  |
| success rate | 待测 |  |
| inference latency | 待测 |  |
| control FPS | 待测 |  |
| peak VRAM | 待测 |  |
| wall-clock time | 待测 |  |

- W&B run：
- TensorBoard log：
- Checkpoint manifest：
- Rollout video：
- Result summary：

## 9. 失败案例

| Case | Observation | Likely cause | Evidence | Next test |
|---|---|---|---|---|
|  |  |  |  |  |

失败类型建议：perception、language grounding、action scaling、coordinate frame、temporal chunk、collision、gripper、latency、distribution shift。

## 10. 问题与排查过程

记录错误信息、最小复现、尝试过的方法，以及哪些假设被排除。不要只记录最终修复。

## 11. 思考

- 结果是否支持假设？
- 模型问题还是系统接口问题？
- 哪个变量最值得做消融？
- 对 Feagine transfer 有什么启示？

## 12. 下一步

- [ ] 待填写
- [ ] 待填写

## 13. 简历材料候选

- 问题：
- 方法：
- 个人贡献：
- 定量结果：
- 工程难点：
- 一句话表述：

只有完成并复核的结果才能进入 `docs/resume.md`。
