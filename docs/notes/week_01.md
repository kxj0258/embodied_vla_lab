# Week 01 Notes

周期：2026-08-25 起。

## 本周目标

完成 Phase 0：建立可复现的 `evla-core` 环境，并理解三个上游仓库的职责和依赖边界。本周不进行正式训练。

## 已完成

- [x] 明确长期项目目标与硬件限制。
- [x] 分析 LeRobot、LIBERO 和 OpenVLA 的关系。
- [x] 收集宿主机 GPU、驱动、OS、conda、磁盘和工具链信息。
- [x] 确定 cu128、双环境和 simulation-first 路线。
- [x] 初始化 `/home/kxj/embodied_ws/embodied_vla_lab` Git 仓库。
- [x] 建立目录与文档骨架。

## 待完成

- [x] 添加三个上游 Git submodule 并记录 commit。
- [x] 阅读 LeRobot `pyproject.toml`、安装文档与 policy/env 入口。
- [x] 阅读 LIBERO environment/evaluation 入口。
- [x] 阅读 OpenVLA inference/action tokenizer/finetune 入口。
- [x] 确定 `evla-core` 候选：Python 3.12、PyTorch 2.11 + cu128；待实测锁定。
- [ ] 创建环境并完成 PyTorch smoke test。
- [ ] 完成 MuJoCo EGL smoke test。
- [ ] 更新 Phase 0 结果与问题记录。

## 当前决策

- 主项目语言：中文。
- GitHub 用户：`kxj0258`。
- Hugging Face 用户：`kxj25`。
- 实验追踪：W&B + 本地 TensorBoard/结构化日志。
- 工作节奏：用户在宿主终端执行命令并反馈；任何安装、下载、训练和 GUI 操作均需明确说明与授权。
- 可投入时间：每天约 3–4 小时。

## 本周风险

- LeRobot 当前依赖与 OpenVLA 历史依赖不能放入同一环境。
- 12 GB 显存不能运行标准 OpenVLA LoRA 配置。
- Codex 沙箱不可直接访问 GPU，验证必须由宿主终端完成。
- 工作盘 126 GB，需要控制数据、模型和缓存数量。

## 本周输出

- `docs/compatibility.md`
- `docs/phases/phase_00_environment.md`
- `docs/architecture.md`
- 第一版项目 README 和 roadmap

## 每日记录

### Day 1

- 完成系统信息采集与架构规划。
- 完成本地 Git 仓库和文档基线。
- 以 shallow submodule 固定三个上游仓库并完成依赖分析。
- 尚未安装依赖、数据集或模型。
