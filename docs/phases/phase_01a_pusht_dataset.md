# Phase 1A — PushT Dataset and ACT Data Contract

更新时间：2026-08-25

状态：完成，尚未提交。

## 目标

固定 PushT Dataset v3 snapshot，验证 episode/sample、视频、statistics、ACT action window、padding、
normalization 和 train/validation split，为 Phase 1B 的小步训练 gate 提供可复现输入。

## 固定数据

| Item | Value |
|---|---|
| repo | `lerobot/pusht` |
| revision | `7628202a2180972f291ba1bc6723834921e72c19` |
| local root | `/home/kxj/.cache/huggingface/lerobot/lerobot/pusht` |
| payload | 8 files / 7,686,801 bytes / 7.5 MiB local `du` |
| Dataset version | v3.0 |
| episodes / frames / tasks | 206 / 25,650 / 1 |
| FPS | 10 |
| image | video HWC `[96,96,3]`; decoded CHW float32 `[3,96,96]` |
| state / action | float32 `[2]` / float32 `[2]` |
| video backend | explicit `pyav` |

完整配置见 `configs/dataset/pusht_v3.yaml`。

## 完成的 gate

- [x] 远端 metadata preflight：完整 SHA、8 files、7,686,801 logical bytes，查询未产生 snapshot。
- [x] 经单独授权下载固定 snapshot；manifest 与远端 metadata 精确一致。
- [x] JSON/Parquet：全局 index、206个 episode 边界、frame index 和10 Hz timestamp 连续。
- [x] DatasetReader 严格离线 PyAV 解码；四个边界样本 shape/dtype/range finite。
- [x] ACT 100-step action window 与 `action_is_pad`：padding 0/39/99/99 精确匹配边界。
- [x] ACT source：delta timestamps、VAE attention mask、L1 mask 和 action queue 语义。
- [x] VISUAL/STATE/ACTION 真实 `MEAN_STD` processor 与 action inverse roundtrip。
- [x] Image stats count 19,619 解释为每 episode 最多100帧的预期抽样，不是缺帧。
- [x] 固定 deterministic length-stratified 185/21 episode split。
- [x] 固化综合脚本并由宿主复跑，`PHASE 1A PUSHT DATASET AUDIT: PASS`。
- [x] 最终文档只读验收：Git 状态符合预期，`git diff --check` 无输出，phase 记录与最新 handoff 一致。

## Split 与第一训练候选

Validation episodes：

```text
[4, 8, 16, 20, 22, 39, 40, 41, 42, 43, 61, 65, 67, 96, 115, 119, 138, 150, 156, 182, 199]
```

- train：185 episodes / 23,043 frames；
- validation：21 episodes / 2,607 frames / 10.1637% frames；
- train/validation mean length：124.5568 / 124.1429；
- chunk50 padded slots：19.6697% / 19.7353%。

Phase 1B 首轮候选：

- `chunk_size=50`：5秒 target horizon；全数据 padded slots 19.6764%；
- `n_action_steps=10`：rollout 每1秒 replan；
- `pretrained_backbone_weights=None`：不下载 ImageNet weights；
- video backend 显式 `pyav`。

这些是训练 smoke 的输入候选，不是训练或控制效果结论。

## 已知例外与失败记录

- `observation.image` stats count 19,619 小于总帧25,650：LeRobot 对每个 episode 最多抽样100帧，
  复算精确一致，归类为预期例外。
- 一次 real-stat processor 命令误把未 collate 的 `[100,2]` action 预期成 `[1,100,2]`，导致 harness
  assertion FAIL；修正为 DataLoader-like batch 后 PASS，roundtrip 最大误差
  `7.62939453125e-06`。
- TorchCodec 在一个 import 顺序下因动态库 ABI 加载失败并 fallback PyAV；DatasetReader-only 路径曾
  通过 TorchCodec。Phase 1 显式固定 PyAV，不安装或修改 TorchCodec/FFmpeg。
- 第一版 split 分析命令排序 validation IDs 后再与 strata `zip`，逐行报告错配；集合和汇总指标未受
  影响。固化脚本保留原 pairing，报告已修正并复核 PASS。
- Dataset `next.success` 全为 false，不能用作 rollout success metric；后续使用环境评测协议。

## 可复现命令

Metadata-only，不下载：

```bash
env -u ALL_PROXY -u all_proxy \
python scripts/fetch_phase1a_pusht_dataset.py \
  --config configs/dataset/pusht_v3.yaml
```

固定 snapshot 已存在后的严格离线综合复核：

```bash
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
HF_HUB_DISABLE_TELEMETRY=1 \
python scripts/check_phase1a_pusht_dataset.py \
  --config configs/dataset/pusht_v3.yaml
```

`fetch_phase1a_pusht_dataset.py --download` 会写入 dataset target，只能在单独获得 dataset 下载授权后
执行。综合复核不联网、不加载模型、不使用CUDA、不训练、不启动GUI、不上传。

## 保存的项目资产

- `configs/dataset/pusht_v3.yaml`：data pin、split 和第一训练候选；
- `scripts/fetch_phase1a_pusht_dataset.py`：metadata-only / 显式下载工具；
- `scripts/check_phase1a_pusht_dataset.py`：严格离线综合 gate；
- `experiments/act_baseline/README.md`：逐 gate 工程记录和后续训练输出要求。

## 尚未完成（Phase 1B）

- 正式训练配置、batch size、BF16、gradient accumulation、optimizer 与 steps；
- 本地 TensorBoard/JSON metric sink 接入 trainer；
- 极小 overfit/smoke、峰值显存和 checkpoint gate；
- 正式训练、loss curve、rollout、success metric、视频和失败分类；
- pretrained ResNet18、W&B、训练、GUI 和上传均继续分别授权。
