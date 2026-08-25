# ACT Baseline

状态：Phase 1A 完成；固定 PushT 数据闭环与可复现脚本已验证，Phase 1B 训练尚未开始。

## 目的

建立项目第一个 image/state → action chunk 模仿学习闭环。先在 PushT 上理解 LeRobot 数据、ACT、normalization 和 rollout，再迁移到一个 LIBERO task。

## 为什么先选 PushT

- 二维连续动作便于检查 action semantics。
- 图像、agent state 和 simulator state 可以分别观察。
- 无语言、无复杂机器人关节和无 MuJoCo 依赖，适合隔离 LeRobot dataset/policy pipeline。
- 96×96 图像和 51.59M ACT 适合 12 GB 单 GPU 做小规模训练。

## Phase 0 已验证接口

### Observation contract

| Key | Runtime shape | Dtype/range | Meaning |
|---|---|---|---|
| raw `pixels` | `[B,96,96,3]` | uint8, 0–255 | PushT top-view RGB |
| `observation.image` | `[B,3,96,96]` | float32, 0–1 | ACT visual input |
| raw `agent_pos` | `[B,2]` | float64 | agent x/y |
| `observation.state` | `[B,2]` | float32 | ACT state input |

### Action contract

| Item | Value |
|---|---|
| dimension | 2 |
| semantics | absolute target position `[x,y]` |
| environment range | `[0,512]` per dimension |
| environment control rate | 10 Hz |
| ACT `chunk_size` smoke config | 100 |
| ACT `n_action_steps` smoke config | 10 |
| training normalization | VISUAL / STATE / ACTION 均为 dataset `MEAN_STD` |

### Model interface result

| Item | Result |
|---|---|
| Backbone | ResNet18 |
| Parameters | 51.59 M |
| Random input batch | image `[1,3,96,96]` + state `[1,2]` |
| Random output | action chunk `[1,100,2]` |
| First-pass latency | 0.282970–0.309598 s，两次接口检查 |
| Peak PyTorch allocation | 222.56–223.43 MiB |
| Status | PASS |

上述延迟包含首次 warm-up，显存仅为 FP32 inference，随机动作没有控制意义。这些数值不进入训练结果或简历定量成果。

## 关键源码链路

```text
PushtEnv / make_env
  → preprocess_observation
  → ACTConfig input_features/output_features
  → ACTPolicy.predict_action_chunk
  → ACT model: ResNet18 + transformer encoder/decoder
  → [batch, chunk_size, action_dim]
```

需要重点理解：

- `lerobot/src/lerobot/envs/configs.py`：环境声明与 Gym kwargs。
- `lerobot/src/lerobot/envs/utils.py`：HWC uint8 到 BCHW float32。
- `lerobot/src/lerobot/policies/act/configuration_act.py`：feature、chunk 和网络超参数。
- `lerobot/src/lerobot/policies/act/modeling_act.py`：action queue、VAE latent 和 action chunk。

## 已知工程问题

Pinned LeRobot 的 PushT 默认 feature 声明为 384×384，但 `gym-pusht==0.1.6` runtime observation 为 96×96。基线固定使用 `configs/env/pusht_96.yaml`，不修改上游 submodule。

## Phase 1A dataset metadata preflight

2026-08-25 对 Hugging Face dataset repo `lerobot/pusht` 的 `main` revision 执行了只读 metadata
查询。查询使用项目既定的命令级代理 workaround：取消 `ALL_PROXY`/`all_proxy`，保留现有
HTTP/HTTPS proxy。

| Item | Result |
|---|---|
| resolved revision | `7628202a2180972f291ba1bc6723834921e72c19` |
| last modified | `2025-09-27 11:40:03+00:00` |
| file entries | 8，全部具有 size metadata |
| known logical size | 7,686,801 bytes / 7.33 MiB |
| video payload | 6,890,970 bytes / 6.57 MiB |
| data parquet | 674,393 bytes / 0.64 MiB |
| metadata | 4 files / 115,481 bytes / 0.11 MiB |
| local cache before/after | 三个候选路径均不存在，未产生 snapshot side effect |
| `/home` free space | 113 GiB |
| result | PASS |

该 gate 只证明远端仓库身份、当前完整 SHA、文件构成、逻辑大小和本地 cache boundary。它没有固定
训练使用的最终 revision、没有下载 dataset，也没有验证 sample、episode、statistics、视频解码、
normalization、action chunk 或 padding contract。后续下载必须固定完整 SHA，并继续单独授权。

### Pinned ACT source preflight

同日只读核对 LeRobot commit `8b256a6c0d4769c3cc3e7e98f04940126398a391` 的 ACT 配置与 loss
源码，结果 PASS：

- 源码默认 `n_obs_steps=1`、`chunk_size=100`、`n_action_steps=100`；Phase 0 smoke 中的
  `n_action_steps=10` 是显式覆盖值，不是上游默认值；
- `action_delta_indices` 为 `range(chunk_size)`，默认对应当前帧起连续 100 个 action；observation
  和 reward delta indices 均为 `None`；
- VISUAL、STATE、ACTION 的默认 normalization mode 都是 `MEAN_STD`，实际数值仍必须由固定
  dataset revision 的 statistics 核对；
- 默认 ResNet18 使用 `IMAGENET1K_V1` pretrained weights，可能触发模型权重下载；Phase 0 smoke
  显式设为 `None`，正式训练前继续作为单独授权 gate；
- 非 temporal-ensemble 推理会在 action queue 为空时预测完整 chunk，只执行前
  `n_action_steps` 个；temporal ensembling 启用时强制 `n_action_steps=1`；
- 训练 L1 loss 使用 `action_is_pad` 反掩码，只对非 padding action 和 action dimensions 求平均；
  VAE/KL 项是否以及如何使用 padding 仍需结合后续模型和 dataset 代码完整核对。

该源码 gate 没有 import policy、没有实例化 backbone、没有下载 pretrained weights，也没有读取或
下载 dataset。

### Delta window and episode padding source gate

只读核对 pinned LeRobot dataset factory、feature utilities 与 reader，结果 PASS：

- policy delta indices 通过 `index / dataset_fps` 转成秒；reader 要求这些秒值在 tolerance 内对齐
  dataset FPS，再通过 `round(delta_seconds * fps)` 还原成整数帧偏移；
- 对当前 PushT 10 Hz contract，ACT 默认 `range(100)` 对应 action timestamps `0.0..9.9` 秒，
  包含当前 action 和未来 99 个 action；
- episode 范围使用 `[dataset_from_index, dataset_to_index)`；超出范围的 query index 会夹到首帧或
  末帧，同时对应的 `<feature>_is_pad` 被设为 `True`；
- 因 ACT 只请求非负 action deltas，`action_is_pad` 默认只会在 episode 尾部出现；被夹到末帧的
  action tensor 只是保持固定 shape 的占位值，不能作为有效训练 target；
- reader 将 padding masks 合并进 sample；对 action feature，键名正是 `action_is_pad`，与 ACT
  training loss 使用的键一致；
- ACT observation delta indices 为 `None`，所以该默认配置没有 observation history window，图像和
  state 只读取当前时刻。

结合上一 gate，当前已确认 L1 loss 排除 `action_is_pad=True` 的位置；VAE encoder 的 attention
padding mask 和 processor 的 normalization/unnormalization 链路仍需继续只读核对。

### ACT processor and VAE padding source gate

只读核对 ACT processor、通用 normalization 实现与 VAE encoder，结果 PASS：

- ACT 直接调用通用 `make_default_pre_post_processors`；preprocessor 对 input 和 output features
  使用同一份 dataset stats，postprocessor 只对 output features 做 inverse transform；
- canonical preprocessor 顺序包含 observation rename、增加 batch dimension、移动到 policy device、
  normalization；postprocessor 对预测 action unnormalize 后移回 CPU；
- `MEAN_STD` forward 为 `(x - mean) / (std + eps)`，inverse 为 `x * std + mean`；由于 forward
  denominator 加了 epsilon，二者在有限精度下不是严格代数逆，预期只有极小差异；
- feature key 完全不在 stats 中时，normalizer 会静默返回原 tensor；key 存在但缺少 `mean` 或 `std`
  时才抛出 `ValueError`。因此正式训练 gate 必须显式核对 image、state、action 的 stats keys、shape、
  finite、std 和 processor 实际使用情况，不能仅凭 config 的 `MEAN_STD` 声明判定已归一化；
- VAE encoder 在 CLS token 和可选 robot-state token 前填入非 padding mask，再拼接
  `action_is_pad` 作为 transformer key padding mask；结合 L1 loss mask，episode 尾部占位 action
  同时不会参与 VAE action-token attention，也不会参与 reconstruction L1；
- PushT contract 含 `observation.state`，所以当前 VAE mask 的 device 来源与项目输入 contract 一致。

该 gate 仍未读取真实 dataset stats，也没有运行 processor roundtrip；这些验证必须在固定 revision
snapshot 经单独授权下载后进行。

### Dataset metadata to policy stats injection gate

只读核对 policy factory 与 normalization processor 的 dataset entry points，结果 PASS：

- 提供 `ds_meta` 时，policy features 由 `ds_meta.features` 转换得到；ACTION features 写入
  `cfg.output_features`，其余 features 在未显式配置时写入 `cfg.input_features`；
- policy factory 将 `ds_meta.stats` 作为 `dataset_stats` 传给 policy，并同时保留 runtime
  `dataset_meta`；
- `NormalizerProcessorStep.from_lerobot_dataset` 与
  `UnnormalizerProcessorStep.from_lerobot_dataset` 均直接使用 `dataset.meta.stats`；
- normalizer 对 transition 中存在的 observation 执行 forward normalization，并在 action 存在且
  类型为 `PolicyAction` 时归一化 action；unnormalizer 对相应输出执行 inverse transform；
- 不提供 dataset metadata 且从环境 feature 随机初始化 policy 时，factory 明确警告 normalization
  缺少 dataset stats。正式 ACT 训练因此必须从固定 PushT revision 的 metadata 建立 feature/stats
  contract，不能沿用 Phase 0 的纯接口 smoke 配置冒充训练配置。

至此，dataset 下载前的 ACT delta window、episode padding、VAE mask、processor 和 stats injection
源码 preflight 已完成。下一 gate 是固定 SHA snapshot 下载，继续需要单独明确授权。

### Fixed-revision PushT snapshot download

获得单独 dataset 下载授权后，使用既定 Hugging Face 命令级代理 workaround 下载固定 revision
`7628202a2180972f291ba1bc6723834921e72c19`，结果 PASS：

| Item | Result |
|---|---|
| destination | `/home/kxj/.cache/huggingface/lerobot/lerobot/pusht` |
| payload | 8 / 8 files |
| payload bytes | 7,686,801 bytes / 7.33 MiB |
| local `du` | 7.5 MiB |
| local-dir metadata | 19 files / 2,786 bytes |
| video | `videos/observation.image/chunk-000/file-000.mp4`, 6,890,970 bytes |
| frame data | `data/chunk-000/file-000.parquet`, 674,393 bytes |
| episode metadata | `meta/episodes/chunk-000/file-000.parquet`, 106,584 bytes |
| extra Hub repo caches | 两个候选路径均未创建 |
| `/home` free space | 113 GiB |

下载 manifest 与 metadata preflight 的文件集合和逻辑字节数精确一致。公共 repo 的 unauthenticated
warning 只影响 rate limit，本次 8/8 文件完整下载不受影响，归为预期提示。该 gate 没有下载模型、
pretrained backbone 或其他 dataset，也没有实例化 `LeRobotDataset`、解码视频或训练。

后续所有检查固定使用上述完整 SHA 和本地目录；下一 gate 先严格离线读取 JSON/Parquet metadata、
feature schema、episode boundary 与 statistics，不先解码视频。

### Offline Dataset v3 metadata and Parquet contract

在 `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1` 下完成 JSON 与 Parquet 只读检查，结果 PASS；没有
解码视频：

| Item | Result |
|---|---|
| codebase version | Dataset v3.0 |
| episodes / frames / tasks | 206 / 25,650 / 1 |
| FPS | 10 |
| published split | 仅 `train: 0:206`，没有预定义 validation split |
| image | video，HWC `[96,96,3]`，10 Hz |
| state | float32 `[2]`，10 Hz |
| action | float32 `[2]`，10 Hz |
| episode lengths | min 49 / median 122 / mean 124.5146 / max 246 |
| global indices | 精确连续 `0..25649` |
| episode indices | 精确覆盖 `0..205`，各 episode 内连续 |
| frame index / timestamp | 每个 episode 从 0 开始，且满足 `timestamp=frame_index/10` |
| episode boundaries | first `[0,161)`；last `[25601,25650)`；metadata 与 frame rows 一致 |

关键 statistics：

- `observation.state` mean `[229.1110,293.3112]`、std `[101.8547,96.4895]`、count 25,650；
- action mean `[228.2396,293.9891]`、std `[101.5996,96.0393]`、count 25,650；
- image mean/std shape `[3,1,1]`，所有值 finite 且 std 为正；image count 为 19,619，与 frame count
  不同，需结合 pinned LeRobot image-stat sampling 实现解释后再归类；
- state 与 action stats 都覆盖完整 25,650 rows，shape 与 ACT feature contract 一致；
- `next.success` 的全局 min/max 均为 false，不能把该字段直接作为后续 rollout success metric；
  正式评测应使用 PushT 环境 reward/success protocol；
- published dataset 只有单一 train split，因此训练/validation episode split 必须由项目显式固定，
  避免按 frame 随机切分造成同一 episode 泄漏。

该 gate 尚未实例化 `LeRobotDataset`、展开 100-step action window、运行 normalization roundtrip 或
解码 AV1 视频；这些继续作为后续分离 gate。

### Image statistics sampling exception

`observation.image` stats count 19,619 与 25,650 frames 不同，已通过 pinned LeRobot 统计源码和
episode lengths 复算，归类为预期抽样语义而非 dataset 缺损：

- image/video statistics 对每个 episode 调用 `estimate_num_samples` 后等距抽样；当前所有 episode
  长度低于约 500，因此每个 episode 的采样数精确为 `min(episode_length,100)`；
- 206 个 episode 的复算抽样总数精确为 19,619，与 `meta/stats.json` 完全一致；
- 其中 153 个长度至少为100的 episode 各采100帧，短 episode 全帧参与；
- image stats 是逐 episode 抽样后的 per-channel `[3,1,1]` 估计，state/action stats 则覆盖全部
  25,650 rows。两类 `count` 的含义不同，不能要求相等。

该 gate 没有解码图像或视频；它只解释已发布 statistics 的生成契约。

### Offline DatasetReader sample and action-window gate

固定 SHA、本地 root、Hugging Face/Datasets 严格 offline 下实例化 `LeRobotDataset`，默认
`torchcodec` 后端成功解码 AV1；四个边界样本全部 PASS：

- dataset length 25,650，runtime revision 精确保持
  `7628202a2180972f291ba1bc6723834921e72c19`；
- decoded image 为 CHW float32 `[3,96,96]`、range `[0,1]`、finite；这与 metadata 的存储声明
  HWC `[96,96,3]` 是 DatasetReader 解码后的预期布局转换；
- state 为 float32 `[2]`，action window 为 float32 `[100,2]`，`action_is_pad` 为 bool `[100]`；
- episode 0 frame 0 padding 0；frame 100 padding 39；episode 0 最后一帧 padding 99；dataset
  最后一帧 padding 99，全部与 `[episode_start,episode_end)` 边界精确一致；
- 尾部 padding tensor 通过重复最后一个有效 action 保持固定 shape，但 mask 精确标记这些占位值；
- task string 为 `Push the T-shaped block onto the T-shaped target.`；
- payload manifest 前后不变；隔离到 `/tmp` 的 datasets cache 已删除；没有实例化模型或训练。

该 gate 将下载后的 storage contract、默认视频解码、单样本和100-step action window 连成了严格离线
闭环。下一步使用真实 stats 验证 ACT preprocessor 的 state/action/image normalization 和 action
postprocessor roundtrip，仍不加载 backbone。

### First real-stat processor attempt — harness shape FAIL

首次 real-stat processor 命令最终触发 `AssertionError`，判为测试脚本 shape 假设错误，不是 dataset
或 normalization 数值失败：

- 输入直接取单个 DatasetReader sample：image `[3,96,96]`、state `[2]`、action window `[100,2]`、
  pad `[100]`；
- 通用 `AddBatchDimensionProcessorStep` 将 image/state 变为 `[1,3,96,96]` / `[1,2]`，但按设计把
  2D action 视为已经 batched，保留 `[100,2]`；pad 也保持 `[100]`；
- 原命令错误断言 action/pad 应变为 `[1,100,2]` / `[1,100]`，因此在 shape assertion 中止；正式
  训练的 DataLoader 会先 collate 为 `[B,100,2]` / `[B,100]`，修正版应显式模拟该 batch boundary；
- 中止前 image/state/action 的真实 stats normalization 已执行，action postprocessor roundtrip 最大
  绝对误差 `7.62939453125e-06`，在预设 tolerance 内；但由于断言提前结束，本 gate 整体仍记 FAIL；
- 本次 import 顺序下 TorchCodec 因 FFmpeg/OpenVINO 动态库链路中的 `CXXABI_1.3.15` 加载问题不可用，
  LeRobot 自动 fallback 到 PyAV，且样本图像成功解码。此前 DatasetReader-only gate 在 TorchCodec
  默认路径下 PASS，说明该问题具有 import-order/runtime-loader 依赖；Phase 1 训练候选配置应显式
  使用已验证的 `pyav` backend，避免依赖该顺序；
- Python 退出时 `TemporaryDirectory` 自动清理，后检没有残留
  `/tmp/evla-pusht-normalization-cache-*`；dataset `du` 仍为 7.5 MiB，Git 状态无额外变化。

下一次重试显式使用 `video_backend="pyav"`，并在调用 processor 前把 sample 的所有 tensor 手工
collate 为 batch size 1，以复现训练 DataLoader contract；不安装或修改 TorchCodec/FFmpeg。

### Batched real-stat processor retry PASS

修正版显式使用 `video_backend="pyav"`，并将单样本手工 collate 为 batch size 1 后，真实 stats
normalization roundtrip 全部 PASS：

| Tensor | Raw shape | Processed shape | Result |
|---|---:|---:|---|
| image | `[1,3,96,96]` | `[1,3,96,96]` | manual MEAN_STD 对照 PASS |
| state | `[1,2]` | `[1,2]` | manual MEAN_STD 对照 PASS |
| action | `[1,100,2]` | `[1,100,2]` | manual MEAN_STD 对照 PASS |
| `action_is_pad` | `[1,100]` bool | `[1,100]` bool | bitwise unchanged，sum 39 |

- action postprocessor 输出保持 `[1,100,2]`、CPU，roundtrip 最大绝对误差
  `7.62939453125e-06`，通过 `atol=1e-5, rtol=1e-6`；
- sample 100 的 normalized state 为 `[-1.2415,-0.7158]`，首个 normalized action 为
  `[-1.3016,-0.6142]`；
- normalized image range 为 `[-7.9391,0.2904]`。全局 image mean 接近白底且部分像素较暗，
  MEAN_STD 实现不 clip，因此大负值是已发布统计量和真实像素共同产生的结果，不是 NaN/Inf；
- 显式 PyAV 避开了上一轮 import-order-dependent TorchCodec loader fallback；当前 Phase 1 训练候选
  video backend 固定为 `pyav`，除非后续有新的独立 TorchCodec 兼容性修复授权；
- 没有实例化 ACT model、加载 pretrained backbone、使用 GPU 或训练。

至此真实 Dataset v3 sample → DataLoader-like batch → ACT preprocessor → normalized action →
postprocessor roundtrip 已闭环。下一步定量比较 chunk sizes 的 padding 成本，并形成 deterministic
episode-level train/validation split 候选。

### Chunk padding and first split candidate analysis

只读统计 206 个 episode 的所有起始帧，结果 PASS；没有写入 split 或训练配置：

| chunk | horizon @10Hz | padded action slots | frames with any padding | mean valid actions |
|---:|---:|---:|---:|---:|
| 25 | 2.5 s | 9.6374% | 19.2749% | 22.5906 |
| 50 | 5.0 s | 19.6764% | 39.3528% | 40.1618 |
| 75 | 7.5 s | 29.6681% | 58.9162% | 52.7489 |
| 100 | 10.0 s | 39.2554% | 75.8908% | 60.7446 |

默认 `chunk_size=100` 对 median 122-step、mean 124.5-step 的数据产生较高 padding：只有6,184个
起始帧拥有完整100-step target。`chunk_size=50` 将 padded slots 约减半至19.68%，仍覆盖5秒预测
horizon，是当前更合理的首轮候选；最终选择仍需与训练 smoke 和 rollout 协议一起固定。

第一版 deterministic 90/10 split 候选使用 seed `20260825` 随机抽21个 validation episodes：

```text
[1, 13, 17, 19, 24, 40, 46, 49, 57, 65, 69, 71, 81, 86, 88, 109, 118, 163, 169, 173, 195]
```

该 split 无 episode overlap，train/validation 为185/21 episodes、23,308/2,342 frames。但 validation
frame fraction 仅9.13%，validation mean length 111.52，低于 train 125.99；因此 chunk100 的
validation padding 43.89% 也高于 train 38.79%。它可复现但长度分布不够匹配，暂不正式采用。

`n_action_steps` 只改变 rollout open-loop duration，不改变训练 target window：1/5/10/20 steps 分别
对应0.1/0.5/1.0/2.0秒。当前保留 `n_action_steps=10` 作为1秒 replanning 候选，不在 Phase 1A
提前声称控制效果。

下一 gate 构造按 episode length 分层且 deterministic 的21-episode validation 候选，并与第一版的
frame fraction、长度分布和 chunk padding 差异对照后再决定 split。

### Deterministic length-stratified split PASS

使用 seed `20260825`，先按 `(episode_length, episode_id)` 排序并切成21个连续 length strata，
再从每个 stratum 固定抽取一个 validation episode。候选结果通过 episode coverage、无 overlap、
frame fraction 和 padding balance 全部断言：

```text
validation episodes:
[4, 8, 16, 20, 22, 39, 40, 41, 42, 43, 61, 65, 67, 96, 115, 119, 138, 150, 156, 182, 199]
```

| Item | Train | Validation |
|---|---:|---:|
| episodes | 185 | 21 |
| frames | 23,043 | 2,607 |
| frame fraction | 89.8363% | 10.1637% |
| length min / median / mean / max | 49 / 122 / 124.5568 / 246 | 56 / 121 / 124.1429 / 190 |
| chunk50 padded slots | 19.6697% | 19.7353% |
| chunk50 frames with padding | 39.3395% | 39.4707% |
| chunk100 padded slots | 39.2521% | 39.2846% |

train/validation mean episode length 只差0.4139 frames；chunk50 padded-slot gap 仅0.0656个百分点。
该 split 比第一版纯随机候选更匹配，正式固定在 `configs/dataset/pusht_v3.yaml`。

原始分析命令的 `STRATA_SELECTION` 明细打印有一个报告 bug：代码先排序 validation IDs，再把它们
与原 strata `zip`，所以逐行显示的 selected episode 不对应该行 stratum。该 bug 不影响实际抽样集合、
train/validation metrics 或断言；固化脚本保留抽样时的 stratum pairing，已修正报告逻辑。

### Phase 1A current decisions

- dataset repo/revision：`lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19`；
- local root：`/home/kxj/.cache/huggingface/lerobot/lerobot/pusht`；
- video backend：显式 `pyav`；
- split：上述 deterministic length-stratified 185/21 episode split；
- normalization：VISUAL、STATE、ACTION 使用固定 snapshot 的 `MEAN_STD` stats；
- first training candidate：`chunk_size=50`、`n_action_steps=10`；这是待 smoke 验证的候选，不是训练结果；
- backbone：继续 `pretrained_backbone_weights=None`，除非后续单独授权下载 ImageNet weights；
- dataset 已下载；模型下载、训练、W&B、GUI、上传仍未授权、未执行。

可复现入口：

```bash
# 默认只查询固定 revision metadata，不写入 dataset target。
env -u ALL_PROXY -u all_proxy \
python scripts/fetch_phase1a_pusht_dataset.py \
  --config configs/dataset/pusht_v3.yaml

# 只有获得单独 dataset 下载授权后才增加 --download。
env -u ALL_PROXY -u all_proxy \
python scripts/fetch_phase1a_pusht_dataset.py \
  --config configs/dataset/pusht_v3.yaml \
  --download

# snapshot 存在后的严格离线综合复核。
python scripts/check_phase1a_pusht_dataset.py \
  --config configs/dataset/pusht_v3.yaml
```

该脚本严格离线，复核 payload、metadata/Parquet、statistics sampling、split/chunk metrics、PyAV
DatasetReader 样本、action padding 和真实 stats processor roundtrip；不实例化 ACT model、不加载
pretrained weights、不使用 GPU、不训练。

### Consolidated reproducibility audit PASS

2026-08-25 由宿主执行固化脚本，完整输出以
`PHASE 1A PUSHT DATASET AUDIT: PASS` 结束。复核结果与此前逐 gate 结果一致：8 files / 7,686,801
bytes、Dataset v3.0、206 episodes / 25,650 frames、固定185/21 split、PyAV、四个 padding 边界样本、
image stats sampling 19,619 和 action roundtrip error `7.62939453125e-06` 全部 PASS。payload manifest
保持不变，脚本明确报告没有网络、模型加载、CUDA、训练、GUI 或上传。

修正后的 stratum report 也验证每一层的 selected episode length 都落在对应 length range 内；先前
一次性分析命令的明细错配仅为打印逻辑 bug，已从固化脚本消除。

Phase 1A 至此关闭。完成结论是 dataset、source contract、split 和 processor 验证，不是训练 loss、
rollout success 或性能 benchmark。

## 计划步骤

- [x] 经授权下载并固定 PushT 当前实验 snapshot。
- [x] 检查 episode/sample 的 key、shape、dtype、FPS 和 episode index。
- [x] 读取 dataset statistics，解释 state/action normalization 与反归一化。
- [x] 核对 ACT action delta indices、padding mask 和 chunk sampling。
- [x] 固定 deterministic episode-level train/validation split。
- [ ] 建立极小 step 的 overfit/smoke 配置，但不自动训练。
- [ ] 通过 gate 后再运行正式单 GPU 训练。
- [ ] 保存曲线、checkpoint manifest、rollout video 和失败案例。
- [ ] 在相同评测协议下迁移到一个 LIBERO task。

## 第一轮训练前必须决定

- [x] Dataset repo ID 与精确 revision。
- [x] 训练/验证 episode split。
- [x] 首轮候选 `chunk_size=50`，根据训练 smoke 保留 go/no-go。
- [x] 首轮候选 `n_action_steps=10`，根据 rollout 保留 go/no-go。
- [x] 首轮候选不使用 pretrained ResNet18；未来权重下载仍需单独授权。
- [ ] batch size、BF16、gradient accumulation 和峰值显存 gate。
- [ ] 极小 overfit/smoke steps、optimizer 和 checkpoint cadence。
- [ ] W&B run name；本地 TensorBoard/JSON metric sink 的 trainer 接入方式。

## 必须输出

- 数据检查报告。
- 训练配置与完整命令。
- loss curve。
- success metric 与评测 episode 数。
- warm-up 后 inference latency/FPS。
- peak training VRAM 和 wall-clock time。
- checkpoint manifest。
- 至少一个成功和一个失败 rollout。
- 对 observation、normalization 和 action chunk 的代码级解释。
