# Embodied VLA Lab — Phase 1A Complete / Phase 1B Planning Handoff

更新时间：2026-08-25

项目根目录：`/home/kxj/embodied_ws/embodied_vla_lab`

## 1. 当前结论

Phase 0 正式提交仍为：

```text
4d67e21 docs: complete phase 0 environment validation
```

Phase 1A 已完成，但本轮 Phase 1A 文件尚未 commit。固定 PushT Dataset v3 snapshot、单样本、视频、
episode/action window、padding、statistics、normalization、split 和综合复现脚本均已验证。

尚未进入 Phase 1B：没有训练配置、训练、checkpoint、rollout、loss 或 success result。

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

## 6. Git 状态与提交边界

`HEAD` 仍为 `4d67e21`。Phase 1A 项目文件均未 commit；提交需要用户单独明确授权。

旧文件 `docs/handoffs/2026-08-25_phase0_complete_phase1a_handoff.md` 是 Phase 1A 开始前的历史快照，
保持原样且仍未跟踪。未来恢复应优先读取本 handoff。

## 7. Phase 1B 尚未完成

- 形成正式 ACT smoke/overfit config；
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
