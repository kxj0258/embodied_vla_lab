# Embodied VLA Lab — Phase 1A Complete / Phase 1B Planning Handoff

更新时间：2026-08-25

项目根目录：`/home/kxj/embodied_ws/embodied_vla_lab`

## 1. 当前结论

Phase 0 正式提交仍为：

```text
4d67e21 docs: complete phase 0 environment validation
```

Phase 1A 已完成并正式提交：

```text
94e2db8 feat: complete phase 1a pusht data validation
```

固定 PushT Dataset v3 snapshot、单样本、视频、episode/action window、padding、statistics、
normalization、split 和综合复现脚本均已验证。

Phase 1B 已进入只读规划与配置 preflight：候选训练配置
`configs/train/act_pusht_phase1b_smoke.yaml` 和离线审计脚本
`scripts/check_phase1b_act_config.py` 已保存，并由宿主完成严格离线验收，最终结果 PASS。
仍未训练，也没有 checkpoint、rollout、loss 或 success result。

## 2. 固定数据与 split

```text
repo: lerobot/pusht
revision: 7628202a2180972f291ba1bc6723834921e72c19
root: /home/kxj/.cache/huggingface/lerobot/lerobot/pusht
payload: 8 files / 7,686,801 bytes / 7.5 MiB du
Dataset: v3.0 / 206 episodes / 25,650 frames / 1 task / 10 Hz
```

Validation episodes：

```text
[4, 8, 16, 20, 22, 39, 40, 41, 42, 43, 61, 65, 67, 96, 115, 119, 138, 150, 156, 182, 199]
```

- train：185 episodes / 23,043 frames；
- validation：21 episodes / 2,607 frames；
- split method：length-stratified、21 strata、seed `20260825`、每层抽一个；
- train/validation mean length：124.5568 / 124.1429；
- chunk50 padded slots：19.6697% / 19.7353%。

固定配置：`configs/dataset/pusht_v3.yaml`。

## 3. 数据与 ACT contract

- stored image：HWC `[96,96,3]` video；decoded image：CHW float32 `[3,96,96]`、`[0,1]`；
- state：float32 `[2]`；action：float32 `[2]` absolute target `[x,y]`；
- ACT 默认 delta indices `range(chunk_size)`；100-step audit window 在10 Hz 下为0.0–9.9秒；
- episode tail query 夹到末 action，同时用 `action_is_pad` 标记；VAE attention 和 L1 loss 均 mask；
- VISUAL / STATE / ACTION 使用 snapshot stats 的 `MEAN_STD`；
- real-stat action inverse roundtrip max error `7.62939453125e-06`；
- `next.success` 全 false，不作为 rollout success metric。

第一训练候选：

```text
chunk_size: 50
n_action_steps: 10
video_backend: pyav
pretrained_backbone_weights: null
```

候选含义：5秒 target horizon、每1秒 rollout replan、无 pretrained weight 下载。尚未训练验证。

## 4. 综合复现证据

宿主运行：

```bash
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
HF_HUB_DISABLE_TELEMETRY=1 \
python scripts/check_phase1a_pusht_dataset.py \
  --config configs/dataset/pusht_v3.yaml
```

最终输出：

```text
PHASE 1A PUSHT DATASET AUDIT: PASS
Dataset payload remained unchanged.
No network, model load, CUDA, training, GUI, or upload was used.
```

保存的工具：

- `scripts/fetch_phase1a_pusht_dataset.py`：默认 metadata-only；`--download` 是显式写入动作；
- `scripts/check_phase1a_pusht_dataset.py`：严格离线综合 audit；
- `configs/dataset/pusht_v3.yaml`：data pin、split、分析与第一训练候选；
- `docs/phases/phase_01a_pusht_dataset.md`：Phase 1A 完成记录；
- `experiments/act_baseline/README.md`：逐 gate 细节。

## 5. 已知例外与失败恢复

- Image stats count 19,619：LeRobot 对每 episode 最多采100帧，复算精确一致，预期例外。
- 一次 processor harness 误把未 collate action `[100,2]` 当作应自动变为 `[1,100,2]`，断言失败；
  改用 DataLoader-like batch 后 PASS，环境干净。
- TorchCodec 在一种 import 顺序下因动态库 ABI 链路失败并 fallback PyAV；显式 PyAV 已验证并固定，
  不安装/修改 TorchCodec 或 FFmpeg。
- 第一版 split 命令的 stratum 明细打印错配，但集合和 metrics 正确；固化脚本修复 pairing 后 PASS。
- Phase 0 三条 OpenCV metadata 例外继续有效，不能安装非 headless OpenCV。
- Phase 1B config/dataset preflight 首次运行在最终 image-stat comparison 处失败：factory 返回
  `numpy.ndarray`，checker 错误调用 `.cpu()`。在此之前 config validate、dataset factory 和固定 split
  断言均已越过；输出目录未创建、临时 cache 已清理、8-file payload manifest 未变。checker 已改为
  `torch.as_tensor(...)` 兼容 NumPy/Tensor；宿主 retry 最终 PASS。
- 同次运行导入 LeRobot 时 TorchCodec ABI loader warning 后回退 PyAV，与 Phase 1A 已知例外一致；
  配置本身显式固定 PyAV，不安装或修改 TorchCodec/FFmpeg。

## 6. Git 状态与提交边界

`HEAD` 为 `94e2db8`，该提交建立在 Phase 0 正式提交 `4d67e21` 上。Phase 1A 项目文件已提交，
提交后的工作树为 clean；没有 push。

旧文件 `docs/handoffs/2026-08-25_phase0_complete_phase1a_handoff.md` 是 Phase 1A 开始前的历史快照，
保持原样并已作为历史记录纳入 `94e2db8`。未来恢复应优先读取本 handoff。

## 7. Phase 1B 尚未完成

已完成第一项只读训练入口核对：

- `TrainPipelineConfig` 默认 `batch_size=8`、`steps=100_000`、`num_workers=4`、DataLoader
  multiprocessing context 为 `spawn`；
- 默认 `env_eval_freq=20_000`、`log_freq=200`、`save_checkpoint=True`、`save_freq=20_000`；
- W&B 默认 `enable=False`（默认 project 名虽为 `lerobot`，但不会单独触发 logger）；
- Accelerator 默认 `mixed_precision="no"`、gradient accumulation steps 为 `1`；其 plugin 固定
  `sync_with_dataloader=False`，防止有限 DataLoader 的 epoch 边界强制提前执行 optimizer step；
- ACT 上游默认 `chunk_size=100`、`n_action_steps=100`、ImageNet ResNet18 pretrained weights、
  AdamW learning rate `1e-5`、weight decay `1e-4`、backbone learning rate `1e-5`，且 scheduler preset
  返回 `None`；AdamW 另有 `betas=(0.9,0.999)`、`eps=1e-8`、`grad_clip_norm=10.0`；项目候选必须
  继续显式覆盖 chunk/action steps 和 pretrained weights；
- 训练循环按 `cfg.batch_size * dp_world_size * gradient_accumulation.steps` 计算 effective batch；
- W&B 仅在 `cfg.wandb.enable` 且配置 project 时创建 logger；checkpoint Hub 上传由独立的
  `save_checkpoint_to_hub` gate 控制；
- LeRobot 使用 draccus 解析 dataclass，训练最小入口是 `dataset.repo_id` 与 `policy.type=act`，所有
  Phase 1B 字段都可通过分层 CLI 路径显式覆盖；官方 ACT 示例中的 `wandb.enable=true`、
  `policy.repo_id` 和后续 rollout 不是必需项，本项目离线 smoke 命令不得照抄这些外部行为；
- `DatasetConfig` 可显式固定 `root`、`revision`、`episodes`、`exclude_episodes` 和 `video_backend`；
  `eval_split` 对每个 task 保留 episode 列表末尾的 `ceil(n * fraction)` 个。PushT 只有一个 task，
  因而直接使用约10%的 split 会选择末尾21个 episode，不会复现 Phase 1A 固定的 length-stratified
  validation IDs；reader 将显式传入的 episode list 原样保存在 `self.episodes`，所以候选配置可将
  185个 train IDs 放在前、21个固定 validation IDs 放在后，再由 `eval_split=21/206` 取尾部；该配置
  仍需离线构造 dataset 后断言实际 split；
- `DatasetConfig.use_imagenet_stats` 默认 `True`，factory 会用 ImageNet mean/std 覆盖 camera stats；
  这与 Phase 1A 已验证的 snapshot image `MEAN_STD` contract 冲突，因此项目候选必须显式设置
  `dataset.use_imagenet_stats=false`；
- `PreTrainedConfig.push_to_hub` 的上游默认值是 `True`，这是 Phase 1B 的关键安全边界：本项目任何
  smoke/overfit 配置都必须显式设为 `policy.push_to_hub=false`，并继续保持
  `save_checkpoint_to_hub=false`；trainer 结束处确实用该 policy flag 进入 model/processor Hub 发布路径；
- policy 默认 `use_amp=False`，Accelerator 默认 `mixed_precision="no"`；两条精度控制路径的实际
  consumption 已区分：training forward/eval 使用 `accelerator.autocast()`，`policy.use_amp` 在本次
  检索中只出现在 control/inference utility；首轮训练 BF16 应通过 accelerator 配置，而不是依赖
  policy flag；
- 本次仅静态读取，没有训练、模型加载、网络访问、输出目录写入或 W&B 行为。

当前仍需：

- 候选仅2 steps、batch 1、BF16、无 checkpoint；config/dataset preflight 已 PASS，未来最小训练
  smoke 仍需用户单独明确授权；
- smoke 通过后再形成 overfit config；
- 决定 batch size、BF16、gradient accumulation、optimizer、LR、steps、checkpoint cadence；
- 选择/实现本地 TensorBoard + JSONL metric sink；
- 只读/静态形成训练命令和输出目录 contract；
- 单独请求训练授权；
- 训练后记录 loss、peak VRAM、wall time、checkpoint、rollout 和 success protocol。

W&B login/run、pretrained ResNet18、训练、GUI、上传继续分别授权。不要自动执行。

## 8. 新对话恢复顺序

1. 本 handoff；
2. `docs/phases/phase_01a_pusht_dataset.md`；
3. `experiments/act_baseline/README.md`；
4. `configs/dataset/pusht_v3.yaml`；
5. `configs/env/pusht_96.yaml`；
6. Phase 0 handoff/compatibility documents（需要环境边界时）。

随后只读核对 Git、submodule pins 和综合脚本/配置存在性。不要重复 Phase 0 或 Phase 1A runtime gate，
除非文件、dataset manifest、environment 或 pinned revisions 发生变化。
