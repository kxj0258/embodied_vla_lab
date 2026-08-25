# ACT Baseline

状态：Phase 1A 完成；Phase 1B 原始ACT的正式100k训练与闭环诊断已完成且success gate未通过；路线B
deterministic ACT 30k训练、checkpoint验证和三checkpoint闭环选择均完成，初步success gate以1/5 PASS。

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

### Phase 1B training-entry static preflight

2026-08-25 只读核对 pinned LeRobot 的 `TrainPipelineConfig` 与 `lerobot_train.py`，结果 PASS：

- trainer 默认 `batch_size=8`、`steps=100_000`、`num_workers=4`，DataLoader multiprocessing
  context 为 `spawn`；这些只是上游默认值，不自动成为本项目 smoke 配置；
- 默认 `env_eval_freq=20_000`、`log_freq=200`、`save_checkpoint=True`、`save_freq=20_000`；
- W&B 默认 `enable=False`；默认 project 字符串 `lerobot` 不会在 enable 为 false 时创建 logger；
- Accelerator 默认 `mixed_precision="no"`、gradient accumulation steps 为 `1`；accumulation plugin
  固定 `sync_with_dataloader=False`，不会在有限 DataLoader 的 epoch 末强制提前同步；
- ACT 上游默认 `chunk_size=100`、`n_action_steps=100`、
  `pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1"`；optimizer preset 为 AdamW，主干与
  backbone learning rate 均为 `1e-5`、weight decay 为 `1e-4`、`betas=(0.9,0.999)`、`eps=1e-8`、
  `grad_clip_norm=10.0`，scheduler preset 为 `None`；
- 项目首轮候选仍显式固定 `chunk_size=50`、`n_action_steps=10`、
  `pretrained_backbone_weights=null`，不能依赖上述上游默认值；
- effective batch size 为 `batch_size * dp_world_size * gradient_accumulation.steps`；
- W&B logger 只有在 `cfg.wandb.enable` 且 project 非空时创建；checkpoint Hub 上传另由
  `save_checkpoint_to_hub` 控制。Phase 1B 配置必须显式关闭两者；
- LeRobot 配置是由 draccus 解析的 dataclass；官方 ACT 训练入口最少指定 `dataset.repo_id` 和
  `policy.type=act`，其他字段可用分层 CLI 路径覆盖。官方示例显式打开 W&B 并设置 policy Hub
  repo，这些不是训练必需项，也不进入本项目的离线 smoke 命令；
- Dataset config 可直接固定本地 `root`、完整 `revision`、episode include/exclude 列表和
  `video_backend`；`eval_split` 按 task 取 episode 列表末尾的 `ceil(n * fraction)` 个。PushT 只有
  一个 task，所以直接设置约10%只会选择末尾21个 episode，不等于 Phase 1A 固定的 length-stratified
  validation IDs；reader 会原样保留显式 episode list，因此候选配置可排列为185个 train IDs 后接
  21个固定 validation IDs，并设置 `eval_split=21/206`。正式采用前仍需通过离线 dataset factory
  断言实际 train/eval IDs；
- `dataset.use_imagenet_stats` 默认 `True`，factory 会覆盖 camera statistics。由于 Phase 1A 已固定
  snapshot image `MEAN_STD`，本项目配置必须显式设置 `dataset.use_imagenet_stats=false`；
- `PreTrainedConfig.push_to_hub` 上游默认竟为 `True`。Phase 1B 所有配置必须显式设置
  `policy.push_to_hub=false`，同时保持 `save_checkpoint_to_hub=false`，不能依赖默认值；trainer
  结束处确实以该 flag 决定是否发布 model 和 processors；
- policy 默认 `use_amp=False`，Accelerator 默认 `mixed_precision="no"`；正式选择 BF16 前需确认
  两条精度开关在当前 trainer 中的职责与交互；当前源码确认 training forward、offline eval 和 env
  eval 使用 `accelerator.autocast()`，而 `policy.use_amp` 在本次搜索范围只用于 control/inference；
  训练 BF16 应配置 Accelerator；
- 本次只读取源码，没有构造 policy、加载模型、使用 CUDA、创建输出目录、训练、联网或触发 W&B。

宿主环境没有 `rg`，首次检索因此中止；使用已有 `grep` 完成同一只读核对，不安装额外工具。

### Reproducibility asset map

此前的宿主命令并非每一段都原样保存为独立 Bash 文件；项目采用“稳定入口合并临时 gate”的方式：

| 宿主工作 | 已保存的稳定入口/记录 | 状态 |
|---|---|---|
| PushT 远端 metadata preflight 与固定 SHA 下载 | `scripts/fetch_phase1a_pusht_dataset.py` + `configs/dataset/pusht_v3.yaml` | 已提交，下载需显式 `--download` |
| payload、metadata、Parquet、image stats sampling | `scripts/check_phase1a_pusht_dataset.py` | 已提交，综合 audit PASS |
| DatasetReader、PyAV、action window/padding | 同一 Phase 1A 综合 audit | 已提交，综合 audit PASS |
| ACT real-stat normalization roundtrip | 同一 Phase 1A 综合 audit | 已提交，综合 audit PASS |
| chunk padding 与 deterministic stratified split | 同一 Phase 1A 综合 audit + dataset config | 已提交，综合 audit PASS |
| 早期随机 split 候选、失败 harness、TorchCodec 例外 | 本 README 与 Phase 1B handoff | 已提交结论；不保留为推荐执行入口 |
| Phase 1B trainer/default/source `sed`/`grep` 核对 | 本 README 与最新 handoff | 结论已落盘；无需依赖易变的一次性检索命令 |
| Phase 1B 2-step BF16 smoke 候选 | `configs/train/act_pusht_phase1b_smoke.yaml` | 已保存、未提交、未训练 |
| Phase 1B config 与精确 split 离线审计 | `scripts/check_phase1b_act_config.py` | 宿主严格离线 retry PASS |

Phase 1A 的推荐复现方式是运行综合脚本，而不是重新拼接对话中的多个 heredoc。原始终端逐行输出目前
只存在于对话/附件中，没有作为 log 文件提交；项目仓库保存的是可执行入口、固定配置、断言、关键
数值、预期例外与 PASS/FAIL 解释。训练开始后应另外保存小型结构化 metrics、checkpoint manifest
与命令摘要，而不是把大段原始控制台日志直接纳入 Git。

当前 Phase 1B config/dataset preflight 的宿主入口：

```bash
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
HF_HUB_DISABLE_TELEMETRY=1 \
WANDB_MODE=disabled \
python scripts/check_phase1b_act_config.py \
  --config configs/train/act_pusht_phase1b_smoke.yaml \
  --dataset-contract configs/dataset/pusht_v3.yaml
```

### Phase 1B config preflight first attempt — harness dtype FAIL

首次宿主运行已成功越过 YAML/draccus parse、`TrainPipelineConfig.validate()`、dataset factory、固定
train/eval episode IDs 和 revision/root 断言，随后在 image-stat comparison 失败：checker 假定 factory
返回 Torch Tensor 并调用 `.cpu()`，实际值是 `numpy.ndarray`。这是审计 harness 的类型假设错误，
不是 config、split、statistics 数值或 dataset 失败。

失败后的边界复核：训练输出目录不存在、`/tmp/evla-phase1b-config-audit-*` 无残留、PushT payload
仍为固定8文件 manifest，Git 只有预期项目文件。checker 已用 `torch.as_tensor(...)` 统一 NumPy/Tensor
表示，等待同一严格离线命令 retry。

导入 LeRobot 时 TorchCodec ABI loader 再次报告已知 warning 并 fallback PyAV；候选 config 与实际
dataset factory 均显式使用 PyAV，因此归为预期例外，不安装或调整 TorchCodec/FFmpeg。

### Phase 1B config/dataset preflight retry PASS

修复 statistics 表示兼容后，宿主以 Hugging Face/Datasets offline 和 `WANDB_MODE=disabled` 重试，
最终输出 `PHASE 1B ACT CONFIG/DATASET PREFLIGHT: PASS`：

- config：2 steps / batch 1 / gradient accumulation 1 / BF16；
- ACT：chunk 50 / execute 10，AdamW preset，无 scheduler；
- split：185 train episodes，validation IDs 与 Phase 1A 固定集合精确一致；
- dataset：固定完整 SHA、PyAV、snapshot image statistics 保持不变；
- safety：W&B、policy Hub push、checkpoint Hub push、env eval、eval recording、checkpoint writing 和
  pretrained download 全部禁用；
- 边界：payload 不变，临时 datasets cache 已删除，输出目录未创建；没有 model、CUDA context、
  training、checkpoint、network、W&B、GUI 或 upload。

TorchCodec loader warning 后 fallback PyAV 仍为已知预期例外，不影响本次 PASS。

### Phase 1B two-step BF16 training smoke PASS

经单独训练授权后，宿主严格离线运行已验收 config，退出码为0。完整控制台输出保存在 Git ignore 的
`results/raw/phase1b_act_pusht_smoke_2step.log`，小型结构化摘要保存在
`results/phase1b_act_pusht_smoke_2step.yaml`。

| Item | Result |
|---|---:|
| train steps / effective batch | 2 / 1 |
| precision | BF16 |
| parameters | 51,563,394 |
| step 1 loss / L1 / KL | 77.405 / 0.767 / 7.664 |
| step 2 loss / L1 / KL | 70.916 / 0.621 / 7.029 |
| step 1 / step 2 wall | 1.067 s / 0.044 s |
| peak PyTorch allocation | 0.94 GiB |
| whole-process wall time | 7.24 s |
| maximum resident set | 2,614,364 KiB |
| checkpoint | none |

logged grad norms `1792.646` 和 `1696.500` 是 `accelerator.clip_grad_norm_` 返回的 clipping 前 norm；
配置的 clipping threshold 是10.0。它们有限且 optimizer step 成功，但绝对值较大，应在 overfit gate
继续观察，而不能仅凭2步判定异常或稳定。

训练结束没有创建 output directory（checkpoint writing 已关闭），Git 仍 clean；`time -v` 报告 socket
messages sent/received 均为0，且 W&B、Hub push、pretrained download、env eval、GUI 均关闭。导入期
TorchCodec ABI warning 后 fallback PyAV 是既有预期例外。

本 smoke 只证明固定 dataset → processor → ACT BF16 forward/backward → gradient clipping → AdamW
step 的最小链路可执行。两个独立 sample 的 loss 下降不构成 convergence、generalization 或 success
证据，也没有产生可供 rollout 的 checkpoint。

### Phase 1B single-episode overfit candidate

2-step smoke 之后的下一 gate 固定为一个轻量、有限的可学习性诊断，而不是正式 benchmark：使用既定
train split 中的 episode 51（73 frames）训练，并使用等长的既定 validation episode 41（73 frames）
观察 held-out loss。二者长度相同，chunk-50 padding 结构一致，可减少 episode length 对比较的干扰。

候选配置为 `configs/train/act_pusht_phase1b_overfit_ep51.yaml`：保持已验证的 batch 1、BF16、chunk 50、
execute 10、AdamW 1e-5、grad clip 10、PyAV 和无 pretrained backbone；训练上限2000步，日志窗口20步，
每100步完整评估73个 held-out frames。只保存最终 checkpoint；W&B、Hub、环境评测、录制和 EMA 关闭。

停止标准：loss/L1/KL/grad norm 出现 NaN/Inf、CUDA OOM、非零退出或数据合同漂移时立即 FAIL；达到
2000步时停止且不自动延长。Overfit PASS 要求最后连续5个日志窗口的平均 train L1 不高于0.10，且
相较最初5个窗口至少下降80%。Validation loss 只用于记录泛化边界，不是 overfit PASS 必要条件。
训练后的 checkpoint reload、固定输入 finite action chunk 和 manifest SHA-256 是独立 gate。

该配置和 `scripts/check_phase1b_act_overfit_config.py` 均为惰性资产；创建与离线审计不构成训练授权。

### Phase 1B single-episode overfit result — execution PASS / gate NO-GO

宿主严格离线完成2000步并以 status 0 退出；最终 checkpoint 与 `last` symlink 完整生成。训练执行、
数值稳定性和 checkpoint writing 判为 PASS，但预先固定的 overfit 门槛未满足：首5个20-step窗口的
train L1 均值为0.5182，末5窗口为0.1584，下降69.43%；最低单窗口 L1 为0.148。末5窗口没有达到
`<=0.10`，下降比例也没有达到80%，因此本 gate 记为 NO-GO，不能自动延长训练或宣称充分记忆。

20次 held-out eval 从 step 100 的1.0164降到 step 2000的0.6604，最低为 step 1300的0.6051；它只说明
等长 validation episode 的 loss 较初期改善，不是 rollout 或 success 证据。Pre-clip grad norm 从首窗口
817.961降至末窗口167.490，全程 finite；PyTorch peak allocation 保持0.94 GiB。完整结果与关键 artifact
哈希见 `results/phase1b_act_pusht_overfit_ep51_2000step.yaml`。

终端自动关闭不是训练故障：启动 wrapper 最后一行使用了 `exit "$train_status"`，成功结束后退出了整个
交互 shell。后续宿主命令不再附带 `exit`，改为只打印 status，避免再次关闭终端。

### Phase 1B overfit checkpoint reload/inference PASS

Step-2000 checkpoint、保存的 preprocessor/postprocessor 和 episode 51 local index 0 均在严格离线下
从本地成功加载。单次 CUDA inference 的 normalized 与反归一化 action chunk shape 都是 `[1,50,2]`，
所有值 finite；反归一化 action range 为 `[11.840057,350.566895]`，首 action 为
`[85.241943,277.314148]`。本次 PyTorch peak allocation 为222.39 MiB，与 Phase 0 单次 ACT inference
量级一致。

原始日志为12行/601 bytes，SHA-256
`a0972d4907ecf06f64eb0c869927ea1d406ddd5a5e7c66ceaaff28b5dae26958`。该 PASS 只证明 checkpoint 与
processor 可加载并能对一个固定 observation 产生有限 action chunk；没有比较 target accuracy、运行
环境 rollout 或产生 success/generalization 结论。

### Phase 1B checkpoint fixed-data accuracy PASS

使用 checkpoint 的部署推理路径遍历 episode 51和等长 validation episode 41的全部73个起点，并仅对
非 padding targets 计算误差。Train/validation full-chunk MAE 分别为17.1033/76.7296 pixels，比例
0.2229；实际会执行的前10步 MAE 为20.7338/58.5615，首动作 MAE 为19.9984/51.9059。训练 episode
显著优于 held-out episode，证明模型确实产生了单 episode 记忆，而不只是训练 loss 的表面变化。

但17.10-pixel train MAE 仍然不可忽略，与预先设定的 overfit NO-GO 一致。Train预测的7300个 action
标量中有15个落在 `[0,512]` 外，最低为-4.9169；validation 预测没有越界。越界统计包含 padding位置，
而所有 MAE 都排除了 padding，因此不能把15个值直接解释为15个将被执行的非法动作。该检查不包含
环境动力学、闭环 replanning 或 success。原始日志35行/1135 bytes，SHA-256
`ee2ce7a45a57a08b7535fb899b35a95467f8bd9bc22129ec41c788bcd8644272`。

### Phase 1B full-split batch-8 resource-gate preflight PASS

单 episode 诊断之后，正式训练前的最后一个资源 gate 固定为完整185/21 split、batch 8、BF16、100步。
配置 `configs/train/act_pusht_phase1b_fullsplit_batch8_100step.yaml` 与离线 checker
`scripts/check_phase1b_act_fullsplit_batch8_config.py` 已通过严格离线审计：train/validation 为
185/21 episodes、23,043/2,607 frames，chunk 50、execute 10、AdamW 1e-5、无 scheduler。

该 gate 每10步记录一次指标，仅在 step 100评估固定 validation 前256 frames，且不保存 checkpoint；
W&B、Hub、pretrained、环境评测、录制和 EMA 均关闭。100步共处理800个训练样本，只相当于完整训练集
0.0347 epoch，目标仅为验证 batch-8 显存、吞吐和数值稳定性，不用于 convergence 判断。Preflight
没有实例化模型、初始化 CUDA、训练、联网或创建 output directory；TorchCodec→PyAV 与无 GPU 审计
环境下 config runtime device 临时回退 CPU 均为已知预期提示，YAML 的 CUDA/BF16 意图由静态断言锁定。

### Phase 1B full-split batch-8 100-step resource gate PASS

宿主从 `(unitree_rl)` 提示符执行命令，但命令内 `conda activate evla-core` 正确切换环境，训练结束提示符
也为 `(evla-core)`。完整185/21 split 的100步 batch-8 BF16 训练和一次256-frame capped eval 均完成，
exit status 0；无 NaN/Inf/OOM、socket、checkpoint 或 output directory。

首/末10-step窗口的 total/L1/KL loss 分别为35.916/0.832/3.508和4.486/0.597/0.389；pre-clip
grad norm 从668.624降至125.571，全部 finite。Step-100 eval loss 为0.4841。排除首个 warm-up窗口后，
平均 step time 为0.0602秒、吞吐132.78 samples/s；PyTorch peak allocation 仅0.94 GiB，因此12 GiB
GPU上的 batch 8资源 gate PASS，无需为了占满显存继续扩大 batch。

全进程 wall time 13.43秒包含初始化和一次 capped eval。基于实测训练吞吐，100k steps 的纯训练约
100分钟；加入20次完整 validation、4次 checkpoint 和启动开销后，正式训练暂估1小时45分到2小时。
这是规划值而非正式 benchmark。结构化结果见
`results/phase1b_act_pusht_fullsplit_batch8_100step.yaml`；raw log 为413行/21,410 bytes，SHA-256
`874cbd7c5fcbf23f07c7e0c1a72ff58df986f353f3afcf93fe1426864e9fcf26`。

### Phase 1B formal 100k config preflight PASS

正式候选 `configs/train/act_pusht_phase1b_formal_100k.yaml` 固定100,000 steps、batch 8，即800,000
samples / 34.7177 epochs；每200步记录、每5,000步完整评估2,607 validation frames，每25,000步保存，
预期 checkpoint steps 为25k/50k/75k/100k。W&B、Hub、pretrained、环境评测、录制和 EMA 继续关闭。

`scripts/check_phase1b_act_formal_config.py` 将其与已通过的100-step resource gate 逐字段比较，只允许
output/job、steps、log/eval/save cadence 和 checkpoint enable发生变化。首次 preflight 因 checker 在
`draccus.parse()` 前漏导入 `ACTConfig`，导致 `act` choice 尚未注册而 FAIL；这是 harness import-order
错误，发生在 dataset factory 之前，无任何 output/cache/model/CUDA/training 副作用。补充显式导入后
同一严格离线命令 retry PASS，确认185/21 split、23,043/2,607 frames、固定 revision 与全部安全边界。

### Phase 1B formal 100k training execution PASS

宿主在严格离线、W&B/Hub关闭的边界内完成100,000个 batch-8 BF16 optimizer steps，两个退出状态均为0。
训练覆盖800,000 samples / 34.7177 effective epochs；500个200-step训练窗口和20次完整2,607-frame
validation eval齐全，没有 NaN/Inf、OOM 或非预期异常。四个计划 checkpoint 均完整生成，`last` 指向
`100000`。

首/末5个训练窗口均值的 total/L1/KL 为 `3.2306/0.5052/0.2724` 和
`0.1682/0.0966/0.0070`，L1下降80.88%。Pre-clip grad norm 从首窗口166.212降到末窗口6.516；
500个窗口中161个高于clip threshold 10、339个不高于10；最后一次超阈值发生在第169个窗口，
其后的331个窗口连续保持不高于阈值。这里记录的是裁剪前范数，不能解释为optimizer实际使用了
同样大小的更新。

完整 validation loss 从step 5,000的0.4093降到step 100,000的0.3819，后者也是20次eval最低值；
首/末5次eval均值为0.40634/0.38922。改善幅度有限且中间有波动，因此只能支持offline held-out loss
有所改善，不能推出环境success、闭环控制质量或跨分布generalization。

500个训练窗口报告的平均step time为0.064736秒、平均吞吐124.16 samples/s；整体wall time为
1:52:38，包含20次完整eval、四次checkpoint、初始化与结束开销。trainer报告的peak PyTorch GPU
allocation为0.94 GiB；`time -v`的maximum RSS为2,883,092 KiB，是主机进程内存口径，不是GPU显存。
socket messages sent/received均为0。

结构化结果见 `results/phase1b_act_pusht_formal_100k.yaml`，独立checkpoint manifest见
`results/phase1b_act_pusht_formal_100k_checkpoint_manifest.yaml`。raw log为930行/4,093,821 bytes，
SHA-256为`3a06879d01e7dd9e3f0bbcad283ec9662ed561f58019b34a8be5ca72436394cd`；较大的字节数来自
carriage-return进度条帧，不是额外训练记录。

### Phase 1B formal checkpoint reload/fixed-data accuracy PASS

最终step-100,000 checkpoint的model SHA-256与manifest精确一致，并在严格离线下成功加载policy与保存的
processors。CUDA deployment-path inference遍历train episode 51和held-out validation episode 41的
全部73个起点，预测均finite；两个episode各7,300个预测标量均在`[0,512]`内。Peak PyTorch allocation
为222.47 MiB，audit exit status为0。

| Metric | Train ep.51 | Validation ep.41 |
|---|---:|---:|
| full chunk MAE | 11.3196 px | 30.8091 px |
| first 10 MAE | 11.5630 px | 20.7707 px |
| first action MAE | 10.8186 px | 11.4301 px |

相对先前step-2,000单episode overfit checkpoint，同一检查中的train full-chunk MAE下降33.82%，
validation full-chunk MAE下降59.85%；validation首10步与首动作MAE分别下降64.53%和77.98%。这说明
full-split正式训练带来了更好的两集固定数据预测，而不仅是训练loss降低。Train/validation full-chunk
ratio为0.3674，但不能把该比例解释为generalization score。

该PASS只覆盖一条train与一条held-out episode，不是全部21条validation episodes的accuracy统计，也未
执行环境动力学、闭环replanning或success判断。原始日志为35行/1,132 bytes，SHA-256为
`e10d01b95bd04147e80396f0153fd15cb99c9d448da7003174187d822e4a2d51`；完整数值已并入
`results/phase1b_act_pusht_formal_100k.yaml`。

### Phase 1B five-episode closed-loop demo — execution PASS / success FAIL

最终100k checkpoint在严格离线、单同步环境、固定seeds `20260825..20260829` 下完成5个PushT闭环
episodes，每集最多300 steps；评测进程、checkpoint接入、processor、环境step、指标聚合和5个本地MP4
均PASS，exit status为0。5集平均sum reward为132.6601，平均max reward为0.62537，但success为0/5，
因此任务success gate严格记FAIL，不能展示成成功策略。

最佳episode 3（seed `20260828`）的sum/max reward为232.4367/0.998760。`gym-pusht==0.1.6`以
`coverage > 0.95`作为success，并将reward定义为clip后的`coverage/0.95`；该峰值对应约94.8822%覆盖，
只差约0.1178个百分点但仍未越过严格阈值。视频显示T块一度几乎贴合目标，随后发生回退/偏移，所以它
适合作为诚实的near-miss demo，而不是success案例。Episode 2达到max reward 0.80275；其余轨迹能接触
和推动T块，但方向、姿态修正或后期稳定性不足。

完整逐seed指标和视频SHA-256见`results/phase1b_act_pusht_demo_5ep.yaml`。该5集诊断证明模型学到了
有意义的闭环推动行为，也暴露offline loss/MAE改善并不自动转化为稳定环境success。由于项目主线将
转入LIBERO/VLA，不继续为PushT做大规模调参或50-seed benchmark。

### Phase 1B replanning/temporal-ensemble diagnostic — inference changes do not fix control

为检验“10-step/1秒open-loop过长”假设，使用同一最终checkpoint、相同5个seeds和相同300-step horizon，
仅将推理`n_action_steps`从10改为5和1。三组执行均PASS，但全部success 0/5：

| Execute steps | Replan interval | Avg sum reward | Avg max reward | Success |
|---:|---:|---:|---:|---:|
| 10 | 1.0 s | 132.6601 | 0.62537 | 0/5 |
| 5 | 0.5 s | 104.6825 | 0.52272 | 0/5 |
| 1 | 0.1 s | 64.5087 | 0.21664 | 0/5 |

相较exec-10，exec-5的平均sum/max reward下降21.09%/16.41%；exec-1下降51.37%/65.36%，评测时间
增加35.28%。Exec-5仅改善seeds 20260825和20260826，却使原先较强的另外3个seeds退化；exec-1没有
在任何seed成为最佳设置，视频中常见接近或轻微接触后缺少持续推动。

因此原主假设不成立：当前失败不能通过单纯提高replanning频率修复。结果反而说明ACT chunk后续动作
承载了持续、连贯的接触轨迹，频繁丢弃后续动作并重新采样第一动作会破坏连续性。剩余瓶颈更符合
精细姿态修正与恢复状态的数据覆盖不足、单帧部分可观测性和behavior-cloning distribution shift。
完整对照见`results/phase1b_act_pusht_replanning_diagnostic_5ep.yaml`。PushT至此冻结，不追加训练或
大规模seed benchmark，项目转入LIBERO语言条件VLA。

最后补测ACT原始`temporal_ensemble_coeff=0.01`：每步预测重叠50-step chunk并融合，而不是裸exec-1
直接丢弃旧chunk。该组执行PASS但仍success 0/5，平均sum/max reward为43.3472/0.30802。相较裸exec-1，
平均max reward提高42.18%，表明ensemble确实缓解了部分瞬时不连贯；但sum reward下降32.80%，未形成
持续有效推动。相较exec-10，sum/max reward分别下降67.32%/50.75%，原seed-20260828 near-miss也从
0.998760退化到0.341267。

因此可以排除“缩短执行chunk”以及“补回原始ACT temporal ensembling”作为主要修复。推理平滑不是
核心瓶颈；失败原因进一步集中到训练数据缺少恢复/精细姿态修正状态、单帧部分可观测性、多模态动作
表示与普通behavior cloning的闭环分布偏移。该结果关闭PushT推理调参分支。

### Phase 1B checkpoint selection diagnostic — execution PASS / all checkpoints success FAIL

为检验最终checkpoint是否掩盖了训练中途更好的闭环策略，对25k/50k/75k checkpoint复用100k的
exec-10协议、相同5个seeds和300-step horizon。加上已有100k结果，四组执行均PASS，但20条受控轨迹
仍全部未成功：

| Checkpoint | Avg sum reward | Avg max reward | Success | 结论 |
|---:|---:|---:|---:|---|
| 25k | 140.9775 | 0.76143 | 0/5 | 最佳aggregate |
| 50k | 76.3645 | 0.29088 | 0/5 | 最差，明显退化 |
| 75k | 99.0276 | 0.51880 | 0/5 | 部分恢复 |
| 100k | 132.6601 | 0.62537 | 0/5 | 第二，包含最接近成功的单条轨迹 |

25k相对100k的平均sum/max reward分别高6.27%/21.75%，说明只按最低offline validation loss选择
100k并不可靠，闭环checkpoint selection确实是一个实质因素。但50k先大幅退化、75k和100k再恢复，
不符合简单的单调overfitting叙事；更准确的结论是闭环质量沿训练轨迹明显非单调，offline objective与
环境success存在错位。

视频抽查进一步显示：25k的seed 20260826一度达到高重合姿态，随后又把T块推离；seeds 20260828和
20260829则停在有少量平移或旋转误差的近目标姿态。75k的seed 20260825也出现“短暂接近、随后退化”。
因此早期checkpoint改善了到达目标附近的能力，却仍缺少越过严格95% coverage阈值所需的终局精调与
状态保持。100k seed 20260828的0.998760 max reward仍是四组最接近成功的单条near-miss。

该对照排除了“成功策略藏在已保存的早期checkpoint里”。它支持训练数据覆盖、BC闭环分布偏移和
offline-to-online model selection共同导致失败，但不能凭5个复用seeds区分各因素的因果占比，也不能
将20条相关轨迹当作20个独立benchmark样本。完整资产见
`results/phase1b_act_pusht_checkpoint_selection_5ep.yaml`；至此冻结PushT，不继续checkpoint sweep或
训练调参，主线转入LIBERO语言条件VLA。

### Route B deterministic ACT 30k ablation pre-registration

路线B只检验ACT VAE训练—部署latent差异这一项核心假设：相对正式100k基线，唯一核心模型变化固定为
`policy.use_vae=false`。候选 `configs/train/act_pusht_phase1b_deterministic_30k.yaml` 保持同一固定
dataset revision、185/21 episode split、batch 8、BF16、chunk/action steps 50/10、无pretrained
backbone、AdamW `lr=1e-5` / weight decay `1e-4` / grad clip `10`、无scheduler和seed `20260825`。

预算固定30,000 steps（240,000 samples / 10.4153 effective epochs），每5,000步完整offline eval，
checkpoint固定为10k/20k/30k。规划训练时间30–40分钟，包含三个checkpoint的固定5-seed exec-10闭环
选择后，总gate约1–1.5小时。这些是规划值，不是本轮实测结果。

`scripts/check_phase1b_act_deterministic_30k_config.py` 对formal 100k YAML做递归叶字段比较，只接受
预注册的 `policy.use_vae`、steps、save cadence、output/job差异；eval cadence虽然属于允许设计维度，
本轮仍保持5k不变。checker还解析两份runtime config，确认baseline/candidate的`use_vae=true/false`，
实例化本地dataset split并断言185/21与23,043/2,607，同时验证payload、输出目录和CUDA均无副作用。
结构化预注册见 `results/phase1b_act_pusht_deterministic_30k_preflight.yaml`。

停止标准预注册如下：非零退出、NaN/Inf、OOM、合同漂移、计划eval或checkpoint缺失立即FAIL；后续对
10k/20k/30k各跑相同5 seeds与exec-10，任一checkpoint至少1/5 success则仅形成初步成功demo并另行
决定是否扩评；三者全部0/5则立即终止PushT，不延长训练、不改学习率/seed/预训练权重。reward改善但
0/5 success只能记诊断改善。配置、preflight与命令均为惰性资产，训练继续需要单独明确授权。

首次preflight在候选raw YAML差异检查之后、dataset factory之前出现预期harness例外：checker对已完成
formal baseline调用`validate()`，其`resume=false`保护按设计拒绝已存在的历史输出目录。这不是候选
合同漂移，且未触发model/CUDA/training/checkpoint/network副作用。修正为baseline只解析ACT默认值、
仅validate新候选后，使用同一严格离线命令重试。

修正后的严格离线retry以exit status 0结束，preflight判定PASS。递归对照精确得到5个预注册叶字段差异：
`job_name`、`output_dir`、`policy.use_vae`、`save_freq`、`steps`；没有额外漂移。runtime解析确认formal
默认`use_vae=true`而候选为false，dataset factory确认固定185/21 split与23,043/2,607 frames。候选
output不存在、dataset payload不变、CUDA未初始化。TorchCodec ABI warning后fallback显式PyAV，以及
无GPU审计环境把runtime device临时回退CPU，均为既有预期提示；YAML的CUDA/BF16意图由静态断言锁定。

唯一候选宿主训练命令如下；它只供用户另行明确授权后执行，本次preflight没有执行：

```bash
cd /home/kxj/embodied_ws/embodied_vla_lab &&
conda activate evla-core &&
test ! -e outputs/train/phase1b_act_pusht_deterministic_30k_bf16 &&
test ! -e results/raw/phase1b_act_pusht_deterministic_30k.log &&
set -o pipefail

/usr/bin/time -v \
env -u ALL_PROXY -u all_proxy \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 \
  WANDB_MODE=disabled \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONUNBUFFERED=1 \
  lerobot-train \
  --config_path=configs/train/act_pusht_phase1b_deterministic_30k.yaml \
  2>&1 | tee results/raw/phase1b_act_pusht_deterministic_30k.log

train_status=${PIPESTATUS[0]}
sha256sum results/raw/phase1b_act_pusht_deterministic_30k.log
wc -l -c results/raw/phase1b_act_pusht_deterministic_30k.log
printf 'TRAIN_EXIT_STATUS=%s\n' "$train_status"
```

### Route B deterministic ACT 30k training — execution PASS / success pending

宿主严格离线完成30,000个batch-8 BF16 optimizer steps，trainer与`time -v`退出状态均为0。150个
200-step训练窗口、6次完整validation eval和10k/20k/30k三个checkpoint均齐全，`last`指向030000；
未发现NaN/Inf、OOM、Traceback或socket消息。训练执行与数值稳定性判PASS，但尚未运行预注册闭环选择，
所以success gate仍为PENDING，不能声称策略成功。

首/末5个窗口的L1均值从0.5238降至0.1710，下降67.35%；pre-clip grad norm均值从38.4094降至
8.3814。150个窗口中67个高于clip threshold 10，83个不高于10，最后一次超阈值是第72个窗口；之后
78个窗口连续不高于10。所有值finite。由于deterministic ACT没有KL项，其total loss等于L1，不能把
该total loss直接与含KL的VAE baseline total loss比较。

offline validation loss依次为`0.4084/0.4083/0.4114/0.3971/0.3914/0.3982`，最低点在25k而非30k；
相同30k预算下，formal VAE baseline的25k最低/30k最终eval为0.4007/0.4131，deterministic候选对应
0.3914/0.3982，分别低2.32%/3.61%。这只支持offline目标小幅改善且checkpoint质量非单调，不能推出
closed-loop reward或success。

训练窗口平均step time为0.05546秒、吞吐144.37 samples/s、peak PyTorch allocation 0.62 GiB；相较
formal VAE baseline前30k的129.20 samples/s与0.94 GiB，吞吐提高约11.74%，allocation降低约34.04%。
全进程wall time为29:14.13，比预估30–40分钟略快；maximum RSS 2,655,952 KiB是主机内存而非GPU显存。

raw log为564行/1,191,295 bytes，SHA-256为
`8fb1b05a16c254687578f8256097a0569ee89060641f387e5386e8c3dcce9ad6`。完整结构化结果与checkpoint哈希见
`results/phase1b_act_pusht_deterministic_30k.yaml`和
`results/phase1b_act_pusht_deterministic_30k_checkpoint_manifest.yaml`。下一gate是严格离线reload与
固定数据accuracy；它只验证三个checkpoint可加载及固定输入输出，不执行环境rollout，后续闭环选择仍
需要单独命令和结果判定。

### Route B deterministic ACT 30k checkpoint reload/fixed-data accuracy PASS

Step-30k模型SHA-256与manifest精确一致，并从本地成功加载policy及processors。CUDA部署推理遍历train
episode 51与validation episode 41各73个起点，所有预测finite，audit exit status为0，peak PyTorch
allocation为155.98 MiB。Raw log为35行/1,139 bytes，SHA-256为
`7fd515210b110e7344e4d391f583f12cf35adc18784409b473a5251aa0a49433`。

| Metric | Train ep.51 | Validation ep.41 |
|---|---:|---:|
| full chunk MAE | 19.7514 px | 29.0490 px |
| first 10 MAE | 17.9099 px | 21.9108 px |
| first action MAE | 14.7366 px | 13.0429 px |

相对formal VAE 100k checkpoint，deterministic 30k的validation full-chunk MAE低5.71%，但validation
first-10与first-action MAE分别高5.49%和14.11%；train三项分别高74.49%、54.89%和36.22%。因此固定
数据对照是mixed result，不能说deterministic模型全面更准确。Train的7,300个预测标量中2个落在
`[0,512]`外，最低-1.102875；该统计包含padding位置，不能直接解释成2个将执行的非法动作。

该PASS只证明30k checkpoint reload、finite deployment inference和两条固定episode target accuracy；
10k/20k尚未单独reload，也未覆盖完整validation、环境reward或success。按预注册协议，下一步先对
10k checkpoint执行相同5 seeds、同步单环境、exec-10闭环评测；收到结果并判定后再推进20k和30k，
避免一次命令跨越多个独立结果gate。

### Route B deterministic ACT 10k closed-loop — execution PASS / success FAIL

10k checkpoint按预注册协议完成固定seeds `20260825..20260829`、同步单环境、每集300 steps、exec-10
的五条轨迹。Eval与`time -v` exit status均为0，五个本地MP4和`eval_info.json`完整生成，socket收发为0；
因此rollout执行PASS。但success为0/5，checkpoint-level任务gate严格FAIL。

平均sum/max reward为85.7531/0.51645。逐seed max reward为
`0.65937/0.93315/0.24989/0.35519/0.38463`；最佳seed 20260826的0.93315对应约88.65% peak
coverage，仍低于严格`coverage > 0.95`门槛，不属于near-success。相较原VAE 25k的平均sum/max reward，
分别低39.17%/32.17%，但25k训练步数是10k的2.5倍，这不是严格的单变量因果对照。

纯eval时间12.21秒、2.442秒/episode，即0.4095 episodes/s；全进程wall time 17.21秒。Maximum RSS
2,268,432 KiB是主机内存；本输出没有报告peak GPU allocation，不能从RSS推断显存。训练loss、offline
eval和grad norm不会因本次rollout改变，仍沿用30k训练审计结论，不能从环境reward反推新的loss/grad值。

结构化增量结果见`results/phase1b_act_pusht_deterministic_checkpoint_selection_5ep.yaml`。当前只能判定
10k失败，尚不能触发“三个checkpoint全部0/5则停止”的最终标准；20k与30k仍须用完全相同协议评测。

### Route B deterministic ACT 20k closed-loop — execution PASS / success FAIL

20k checkpoint完成相同五seed同步exec-10协议，eval与wrapper exit status均为0，五个视频与eval_info
齐全、socket为0，因此执行PASS；success仍为0/5，checkpoint-level任务gate FAIL。

平均sum/max reward为81.5999/0.33700，相较10k分别下降4.84%/34.75%，说明增加到20k没有改善闭环，
反而显著降低峰值重合能力。最佳seed仍为20260826，但max reward仅0.57383，对应约54.51% coverage。
Seed 20260829的sum reward仅1.2549、max reward 0.11007，是当前最明显退化轨迹。相较VAE 25k的
平均sum/max reward低42.12%/55.74%；VAE多5k steps，因此该比较仍是描述性参照，不是严格单变量因果证据。

纯eval时间12.80秒、2.560秒/episode，即0.3906 episodes/s；全进程wall time 18.41秒。Maximum RSS
2,260,188 KiB是主机内存，本输出仍没有peak GPU allocation。训练loss/eval/grad/吞吐/显存不受本次
rollout改变，继续以30k训练结构化结果为准。

10k和20k累计0/10 success，但预注册停止标准要求10k/20k/30k全部完成后决定，因此现在不能提前停止；
下一步只评测30k checkpoint。若仍为0/5，将严格触发PushT终止，不追加训练长度、学习率、seed、预训练
权重或推理变体。

### Route B deterministic ACT 30k closed-loop — execution PASS / initial success PASS

30k checkpoint完成相同五seed同步exec-10协议，eval与wrapper exit status均为0，五个视频、eval_info和
raw log完整，socket为0。Seed 20260825明确返回`success=true`且max reward为1.0，因此checkpoint执行
PASS、success为1/5，并严格满足预注册的“任一checkpoint至少1/5 success”初步成功条件。这是本项目
PushT阶段首个真实成功rollout，不是near-miss。

30k平均sum/max reward为127.4630/0.74039，相较10k提高48.64%/43.36%，相较20k提高56.20%/119.70%。
逐seed max reward为`1.0/0.43297/0.97816/0.49127/0.79955`，除成功轨迹外seed 20260827也是接近阈值
但未成功的轨迹。相较原VAE 25k，平均sum/max reward低9.59%/2.76%，但VAE为0/5而deterministic为1/5；
相较VAE 100k，sum低3.92%、max高18.39%，同样由0/5提升到1/5。不同训练步数限制了纯`use_vae`
因果强度，但在共享五seed协议下，deterministic 30k确实首次越过严格success门槛。

纯eval时间12.87秒、2.575秒/episode，即0.3884 episodes/s；全进程wall time 18.32秒。Maximum RSS
2,262,752 KiB是主机内存，本输出没有peak GPU allocation。训练loss/eval/grad/吞吐/显存保持此前记录：
末5窗口L1 0.1710、pre-clip grad 8.3814、144.37 samples/s、0.62 GiB allocation、29:14 wall time。

三checkpoint共15条配对轨迹只有1次成功：10k与20k均0/5，30k为1/5；闭环质量随训练明显非单调。
Offline validation最低在25k，而成功checkpoint 30k的eval loss略回升到0.3982，再次说明offline loss不能
独立选择closed-loop策略。1/5只建立初步成功demo，不能作为稳定20% benchmark success rate或泛化结论。

按预注册规则，达到初步成功后不自动扩评、不追加训练或推理变体。是否扩大seed数用于置信区间、是否
将成功视频作为展示资产、以及是否结束PushT转入LIBERO，都需要下一步单独决定。

### Route B final integrity verification PASS

宿主最终只读复核中`git diff --check`无输出，判定PASS；30k `eval_info.json`、seed-20260825成功视频和
rollout raw log的SHA-256分别为`261fa202...510f`、`d99581...4123`和`838881...2fa`，与结构化结果
逐字节一致。工作树只包含预期的Route B配置、checker、结构化结果、manifest、handoff与README修改，
没有发现意外路径。该核验不产生新的loss/eval/grad/吞吐/显存/时间数据，也没有执行commit或push。

### Route B independent 100-seed paired evaluation pre-registration

为区分“存在一次成功”与“可重复成功率”，后续独立测试冻结100个未参与开发的新seeds
`20260830..20260928`。先评测deterministic ACT 30k，再用完全相同seeds评测原VAE 25k；两者均保持
PushT 300-step horizon、chunk/execute 50/10、batch 1、同步单环境、严格离线。开发用5 seeds不进入
该测试集，执行中不得重选checkpoint、改推理协议、替换seed或提前停止。

主指标预注册为每模型success count/rate与95%二项区间，以及逐seed配对success列联表、success-rate
difference和exact McNemar test。平均/中位sum reward、max reward与配对bootstrap差异为次要指标。
100-seed独立结果将取代开发5-seed的20%点估计用于性能描述；但VAE 25k与deterministic 30k仍有5k
训练预算差，不能称为完美等预算的`use_vae`单因素实验。

当前评测器只保存每模型前10条视频，但在`eval_info.json`记录全部100条逐seed指标。按5-seed实测，
每模型纯eval约4–4.5分钟，两模型连初始化与视频写入预计9–12分钟。完整协议固定在
`results/phase1b_act_pusht_deterministic_vs_vae_100seed_preregister.yaml`；两组评测不授权新训练，
也不自动触发后续模型修改。

### Route B deterministic ACT 30k independent 100-seed result — execution PASS / 6% success

Deterministic 30k在100个未参与开发的连续seeds上完成严格离线评测，eval与wrapper exit status均为0，
全部100条逐seed指标和前10条视频齐全，socket为0。实际seed集合为`20260830..20260929`；预注册文件
最初把末值误写成20260928，这是区间算术文档错误，因为从20260830递增99次应为20260929。运行确实
使用100个全新连续seeds，后续VAE配对组已锁定复用同一实际集合，因此不影响配对有效性。

结果为6/100 success，即6%。成功seeds为
`20260832/20260844/20260854/20260860/20260883/20260888`。Clopper-Pearson exact 95%区间约为
`[2.23%,12.60%]`，Wilson区间约为`[2.78%,12.48%]`。这证明原5-seed中的成功不是唯一偶然状态，
但也把开发集上的1/5=20%明显校正为更低的独立测试可靠性，不能继续用20%描述模型性能。

平均/中位sum reward为72.3539/64.5132，平均/中位max reward为0.51557/0.47643。24条轨迹达到
max reward至少0.90，13条达到至少0.95，但只有6条注册success，说明策略经常接近目标，却仍有显著的
终局精调、阈值跨越或成功状态保持缺口；另有7条sum reward为0，反映部分初始状态下完全失效。

纯eval 116.81秒、1.168秒/episode，即0.8561 episodes/s；全进程wall time 2:03.22，明显快于保守
预估。Maximum RSS 2,262,624 KiB是主机内存，本输出未报告GPU allocation。训练loss、offline eval、
grad norm、训练吞吐和训练显存没有被本次rollout重测或改变，仍以既有30k训练结果为准。

结构化结果见`results/phase1b_act_pusht_deterministic_030000_100seed.yaml`。在VAE 25k使用完全相同
100 seeds完成前，只能报告deterministic的绝对可靠性，不能声称相对VAE的稳定提升。

### Route B independent 100-seed paired comparison — execution PASS / mixed evidence

原VAE 25k在完全相同`20260830..20260929`的100个seeds上执行PASS，得到2/100 success，exact 95%
区间约`[0.24%,7.04%]`；deterministic 30k为6/100，区间约`[2.23%,12.60%]`。点估计上关闭VAE使
success率从2%升至6%，绝对+4个百分点、相对3倍。

逐seed列联表为：共同成功0、仅deterministic成功6、仅VAE成功2、共同失败92。Exact McNemar双侧
`p=0.2891`，配对bootstrap的success差95%区间约`[-1pp,+10pp]`，因此success提升尚未达到统计稳定，
不能表述为已显著优于VAE。两模型成功seed完全不重叠，说明它们覆盖不同初始状态模式。

Deterministic平均max reward为0.51557，VAE为0.41226，绝对提高0.10331、相对提高25.06%；配对
bootstrap 95%区间`[0.02834,0.17757]`不跨0。这是当前最稳定的改善证据，表示关闭VAE较一致地提高了
峰值目标重合度。平均sum reward提高5.03（7.47%），但区间`[-7.62,17.36]`跨0，不构成稳定提升。

Deterministic只有7条zero-sum轨迹，VAE有22条；达到max reward至少0.90的轨迹为24对12，至少0.95
为13对7。这进一步说明deterministic较少完全失效，也更常进入近目标状态，但从近目标转成严格success
仍然困难。

VAE纯eval为139.38秒、1.394秒/episode；deterministic为116.81秒、1.168秒/episode，deterministic
推理约快16.2%。VAE全进程wall time 2:25，maximum RSS 2,277,580 KiB；两组均未报告GPU allocation。
训练loss、offline eval、grad norm、训练吞吐与训练显存没有在此次评测中重测。

最终结论是mixed but supportive evidence：`use_vae=false`提高了成功率点估计、显著改善了配对max reward、
减少灾难性失败且推理更快，但100 seeds仍不足以让稀疏success差异达到统计稳定；加之30k与25k预算差，
不能宣称纯VAE因素已被最终证明。两模型绝对成功率都很低，下一步高收益方向仍是恢复状态与终局精调
数据，而不是继续用同一checkpoint扩seed追逐显著性。

### Route B 100-seed paired evaluation final integrity PASS

宿主最终只读核验中`git diff --check`无输出；deterministic/VAE的eval_info SHA-256分别为
`bee0af3a...e8f7`和`64baef8d...44c`，raw log SHA-256分别为`66a7c4fb...1cd1`和
`ff2add94...406`，全部与结构化结果一致。工作树仅包含预期的100-seed协议、结果、README和manifest
更新，无意外路径；没有执行commit或push，也没有产生新的评测指标。

## 计划步骤

- [x] 经授权下载并固定 PushT 当前实验 snapshot。
- [x] 检查 episode/sample 的 key、shape、dtype、FPS 和 episode index。
- [x] 读取 dataset statistics，解释 state/action normalization 与反归一化。
- [x] 核对 ACT action delta indices、padding mask 和 chunk sampling。
- [x] 固定 deterministic episode-level train/validation split。
- [x] 宿主验收已保存的2-step、batch-1、BF16 smoke 配置及其严格离线审计脚本；未训练。
- [x] 经单独授权完成2-step、batch-1、BF16 training smoke。
- [x] 根据 smoke 结果建立并执行独立 overfit 诊断。
- [x] 完成正式单 GPU 100k训练并保存四个checkpoint。
- [x] 验证正式100k checkpoint reload与两集固定数据accuracy。
- [x] 保存checkpoint manifest、rollout video和跨checkpoint失败案例；loss曲线仍待生成。
- [ ] 在相同评测协议下迁移到一个 LIBERO task。

## 第一轮训练前必须决定

- [x] Dataset repo ID 与精确 revision。
- [x] 训练/验证 episode split。
- [x] 首轮候选 `chunk_size=50`，根据训练 smoke 保留 go/no-go。
- [x] 首轮候选 `n_action_steps=10`，根据 rollout 保留 go/no-go。
- [x] 首轮候选不使用 pretrained ResNet18；未来权重下载仍需单独授权。
- [x] batch 1、BF16、gradient accumulation 1 的 smoke 峰值 PyTorch allocation 为0.94 GiB。
- [x] 正式训练固定100k steps，并以25k cadence保存四个checkpoint。
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
