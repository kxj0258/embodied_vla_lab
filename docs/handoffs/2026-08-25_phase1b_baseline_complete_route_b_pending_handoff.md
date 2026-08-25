# Embodied VLA Lab — Phase 1B Baseline Complete / Route B Pending Handoff

更新时间：2026-08-25

项目根目录：`/home/kxj/embodied_ws/embodied_vla_lab`

这是当前最新 handoff。Phase 1B 原始 ACT PushT 基线的训练、checkpoint验证、闭环评测、推理消融与
checkpoint sweep均已完成。PushT success gate未通过；用户认可仅再执行一次有严格停止条件的路线B：
deterministic ACT（`use_vae=false`）最小消融。路线B尚未创建配置或启动训练。

## 1. Git、submodule 与授权边界

创建本handoff前正式HEAD：

```text
16232e3833d4a225b903ed60f2d7062ebaa68690
16232e3 docs: record phase 1b act smoke result
```

三个submodule pins未改变：

```text
lerobot  8b256a6c0d4769c3cc3e7e98f04940126398a391
libero   8f1084e3132a39270c3a13ebe37270a43ece2a01
openvla  c8f03f48af692657d3060c19588038c7220e9af9
```

用户已授权将当前Phase 1B资产分两次本地commit：

1. `feat: add reproducible phase 1b ACT training pipeline`
2. `docs: finalize ACT PushT closed-loop diagnostics`

本次授权不包含push，也不包含路线B训练。安装、dataset/model下载、任何新训练、GUI、W&B、Hub/upload、
后续commit和push仍需分别明确授权。不要使用多agent。

## 2. 正式100k训练结论

固定配置：

```text
dataset: lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19
split: 185 train episodes / 21 validation episodes
steps: 100,000
batch: 8
effective samples/epochs: 800,000 / 34.7177
precision: BF16
chunk/action steps: 50/10
backbone: ResNet18, no pretrained weights
optimizer: AdamW, lr 1e-5, weight decay 1e-4, grad clip 10
W&B/Hub/env eval/recording/EMA: disabled
```

训练执行PASS：exit status 0，100k steps与20次完整validation eval齐全，25k/50k/75k/100k四个
checkpoint完整，`last -> 100000`。没有NaN/Inf、OOM或非预期异常。

```text
train L1 first/last five-window mean: 0.5052 -> 0.0966 (-80.88%)
eval loss step 5k -> 100k: 0.4093 -> 0.3819
pre-clip grad norm first/last five-window mean: 166.212 -> 6.516
mean throughput: 124.16 samples/s
peak PyTorch allocation: 0.94 GiB
wall time: 1:52:38
raw log sha256: 3a06879d01e7dd9e3f0bbcad283ec9662ed561f58019b34a8be5ca72436394cd
```

结构化资产：

- `results/phase1b_act_pusht_formal_100k.yaml`
- `results/phase1b_act_pusht_formal_100k_checkpoint_manifest.yaml`
- `configs/train/act_pusht_phase1b_formal_100k.yaml`

## 3. 最终checkpoint固定数据accuracy

100k checkpoint严格离线reload与CUDA deployment-path检查PASS。模型SHA-256：
`c9fd17bfe0e909f62b3bb67500e85c46b4c42a75d6edc043508cd5cf22244f01`。

| Metric | Train episode 51 | Validation episode 41 |
|---|---:|---:|
| full chunk MAE | 11.3196 px | 30.8091 px |
| first 10 MAE | 11.5630 px | 20.7707 px |
| first action MAE | 10.8186 px | 11.4301 px |

该PASS只证明固定数据预测与checkpoint链路，不证明闭环success。

## 4. 闭环基线与推理诊断

100k checkpoint在固定seeds `20260825..20260829`、exec-10、每集最多300 steps下执行PASS，但success
为0/5。平均sum/max reward为132.6601/0.62537。最佳seed `20260828`的max reward为0.998760，对应
约94.8822% coverage，仍未越过严格`coverage > 0.95`阈值，只能作为near-miss。

相同checkpoint与seeds的推理消融：

| Setting | Avg sum reward | Avg max reward | Success |
|---|---:|---:|---:|
| exec-10 | 132.6601 | 0.62537 | 0/5 |
| exec-5 | 104.6825 | 0.52272 | 0/5 |
| exec-1 | 64.5087 | 0.21664 | 0/5 |
| exec-1 + temporal ensemble 0.01 | 43.3472 | 0.30802 | 0/5 |

缩短replanning interval与ACT原始temporal ensembling均未修复控制，推理调参分支关闭。

## 5. 四checkpoint选择诊断

25k/50k/75k复用同一exec-10与5-seed协议，加上100k共20条配对轨迹，全部success FAIL：

| Checkpoint | Avg sum reward | Avg max reward | Success |
|---:|---:|---:|---:|
| 25k | 140.9775 | 0.76143 | 0/5 |
| 50k | 76.3645 | 0.29088 | 0/5 |
| 75k | 99.0276 | 0.51880 | 0/5 |
| 100k | 132.6601 | 0.62537 | 0/5 |

25k是最佳aggregate checkpoint，100k包含最接近成功的单条轨迹。50k退化后75k/100k恢复，说明闭环
性能沿训练轨迹非单调；最低offline validation loss不能可靠选择最佳closed-loop checkpoint。视频抽查
显示反复出现“到达近目标后推离”与“残留平移/旋转误差后停滞”。早期checkpoint中没有隐藏的成功策略。

结构化资产：

- `results/phase1b_act_pusht_demo_5ep.yaml`
- `results/phase1b_act_pusht_replanning_diagnostic_5ep.yaml`
- `results/phase1b_act_pusht_checkpoint_selection_5ep.yaml`

## 6. 当前结论边界

- 原始ACT已学到有意义的接触、推动和近目标行为，但未形成稳定终局精调、成功状态保持和偏离恢复。
- checkpoint selection与offline-to-online objective mismatch是实质因素，但不能解释全部失败。
- 五个复用seeds适合受控配对诊断，不构成稳定benchmark置信区间。
- 不再追加原始ACT训练、execution cadence、temporal ensemble或checkpoint sweep。
- PushT只保留一次路线B最小消融机会；失败后立即转入LIBERO语言条件VLA主线。

## 7. 路线B：deterministic ACT预注册方案

核心假设：当前默认ACT使用`use_vae=true`、`kl_weight=10`；训练时posterior latent依赖真实action，部署
时使用zero latent。PushT示范接近单一专家策略，移除VAE可能减少训练—部署latent mismatch。

只允许一个主变量变化：

```text
policy.use_vae: false
```

保持不变：dataset revision、185/21 split、无pretrained backbone、chunk/action steps 50/10、batch 8、
BF16、AdamW、seed、严格offline和W&B/Hub/recording关闭。

候选预算：30k steps，10k/20k/30k checkpoints，每5k offline eval；三个checkpoint各用相同5 seeds做
exec-10闭环选择。预计训练30–40分钟，全部配置、审计和评测约1–1.5小时。

预注册停止标准：

- 任一checkpoint达到至少1/5 success：形成初步成功demo，再单独决定是否授权扩大评测；
- 三个checkpoint全部0/5：立即停止PushT，不追加学习率、seed、预训练backbone或更长训练；
- reward改善但0/5只能记诊断改善，不能宣传为成功策略。

## 8. 下一步严格顺序

1. 只读核对提交后的HEAD、status和submodule pins。
2. 创建路线B 30k配置与严格离线checker；配置/审计不构成训练授权。
3. 运行checker并固定结构化preflight与预注册停止标准。
4. 向用户报告配置差异、预估时间和唯一训练命令。
5. 只有获得新的明确训练授权后才能启动路线B；不得自动训练。

路线B第一阶段不下载ImageNet weights、不修改LeRobot ACT源码、不加多帧观测、不改loss weighting，也不
执行temporal ensemble。这些会混淆`use_vae=false`的因果解释或扩大工程范围。
