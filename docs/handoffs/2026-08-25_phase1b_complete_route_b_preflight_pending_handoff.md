# Embodied VLA Lab — Phase 1B Complete / Route B Preflight Pending Handoff

更新时间：2026-08-25

项目根目录：`/home/kxj/embodied_ws/embodied_vla_lab`

这是下一对话必须首先完整读取的最新handoff。原始ACT PushT Phase 1B基线已经训练、验证、诊断并分两次
本地提交；工作树在创建本文件前clean。用户认可路线B“最小改进版ACT”，但路线B配置、checker和训练
均尚未开始。下一对话先接管上下文，再创建配置与严格离线preflight；不得直接启动训练。

## 1. 当前Git状态

用户在宿主最终复核：

```text
## main
9a589d6 (HEAD -> main) docs: finalize ACT PushT closed-loop diagnostics
0322b1c feat: add reproducible phase 1b ACT training pipeline
16232e3 docs: record phase 1b act smoke result
```

两个本轮提交的职责边界：

- `0322b1c`：训练配置、审计脚本、overfit/resource gate/formal结果、checkpoint manifest与历史handoff；
- `9a589d6`：闭环demo、replanning/temporal-ensemble诊断、四checkpoint选择结果、README与阶段handoff。

没有push。创建本文件后，预期工作树只新增：

```text
?? docs/handoffs/2026-08-25_phase1b_complete_route_b_preflight_pending_handoff.md
```

三个submodule pins未改变：

```text
lerobot  8b256a6c0d4769c3cc3e7e98f04940126398a391
libero   8f1084e3132a39270c3a13ebe37270a43ece2a01
openvla  c8f03f48af692657d3060c19588038c7220e9af9
```

## 2. 授权与操作边界

继续遵守：安装、dataset/model下载、任何新训练、GUI、W&B、Hub/upload、commit和push都需要分别明确
授权。用户本轮授权的两次commit已经完成并消费；不自动延续到本handoff或路线B提交。

同类只读查看可自行执行，使用`evla-core`环境。不得使用multi-agent。不要删除、resume或覆盖已有训练、
checkpoint、raw log和rollout视频。

下一对话可以在不训练的边界内创建路线B配置、checker、preflight结构化资产并运行严格离线checker；
路线B训练必须在preflight PASS后再次取得用户明确授权。

## 3. 固定数据与原始ACT配置

```text
dataset: lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19
local root: /home/kxj/.cache/huggingface/lerobot/lerobot/pusht
dataset size: 206 episodes / 25,650 frames / 10 Hz
train split: 185 episodes / 23,043 frames
validation split: 21 episodes / 2,607 frames
video backend: pyav
training: 100,000 steps / batch 8 / BF16
chunk/action steps: 50/10
backbone: ResNet18, pretrained weights disabled
optimizer: AdamW lr 1e-5, weight decay 1e-4, grad clip 10
seed: 20260825, cudnn deterministic
W&B/Hub/env eval/recording/EMA: disabled
```

正式配置：`configs/train/act_pusht_phase1b_formal_100k.yaml`。

## 4. 正式100k训练结果

执行PASS：100k steps、20次完整validation eval、25k/50k/75k/100k checkpoints全部齐全，`last`指向
100k；exit status 0，无NaN/Inf、OOM、socket或非预期异常。

```text
train L1 first/last five-window mean: 0.5052 -> 0.0966 (-80.88%)
eval loss step 5k -> 100k: 0.4093 -> 0.3819
pre-clip grad norm first/last five-window mean: 166.212 -> 6.516
mean throughput: 124.16 samples/s
peak PyTorch GPU allocation: 0.94 GiB
wall time: 1:52:38
raw log: 930 lines / 4,093,821 bytes
raw log sha256: 3a06879d01e7dd9e3f0bbcad283ec9662ed561f58019b34a8be5ca72436394cd
```

结构化资产：

- `results/phase1b_act_pusht_formal_100k.yaml`
- `results/phase1b_act_pusht_formal_100k_checkpoint_manifest.yaml`
- `experiments/act_baseline/README.md`

## 5. Checkpoint fixed-data accuracy

100k checkpoint严格离线reload和CUDA deployment inference PASS。Model SHA-256：
`c9fd17bfe0e909f62b3bb67500e85c46b4c42a75d6edc043508cd5cf22244f01`。

| Metric | Train episode 51 | Validation episode 41 |
|---|---:|---:|
| full chunk MAE | 11.3196 px | 30.8091 px |
| first 10 MAE | 11.5630 px | 20.7707 px |
| first action MAE | 10.8186 px | 11.4301 px |

该PASS只覆盖两条固定episode，不证明环境success或generalization。

## 6. 100k闭环demo与推理消融

固定seeds `20260825..20260829`、同步单环境、每集最多300 steps：

```text
100k exec-10: avg sum 132.6601 / avg max 0.62537 / success 0/5
100k exec-5:  avg sum 104.6825 / avg max 0.52272 / success 0/5
100k exec-1:  avg sum 64.5087  / avg max 0.21664 / success 0/5
100k exec-1 + temporal ensemble 0.01:
               avg sum 43.3472 / avg max 0.30802 / success 0/5
```

100k seed `20260828`的max reward `0.998760`对应约94.8822% coverage，仍未越过严格
`coverage > 0.95`成功阈值，只能作为near-miss。缩短执行chunk与temporal ensembling均未修复控制，
该推理调参分支已经关闭。

结果：

- `results/phase1b_act_pusht_demo_5ep.yaml`
- `results/phase1b_act_pusht_replanning_diagnostic_5ep.yaml`

## 7. 四checkpoint闭环选择

相同5 seeds和exec-10协议：

| Checkpoint | Avg sum reward | Avg max reward | Success |
|---:|---:|---:|---:|
| 25k | 140.9775 | 0.76143 | 0/5 |
| 50k | 76.3645 | 0.29088 | 0/5 |
| 75k | 99.0276 | 0.51880 | 0/5 |
| 100k | 132.6601 | 0.62537 | 0/5 |

25k为最佳aggregate checkpoint，100k包含最接近成功的单条轨迹。50k退化、75k/100k恢复，说明闭环
性能随训练非单调，最低offline validation loss不能可靠选择最佳closed-loop checkpoint。四checkpoint
共20条配对轨迹全部失败，排除了“成功模型藏在早期checkpoint”的可能。

抽帧观察的重复失败模式：

- 到达高重合姿态后继续推动，最终又离开目标；
- 停在近目标姿态，但残留少量平移/旋转误差；
- 缺少终局精调、成功状态保持和偏离后的恢复。

完整结果：`results/phase1b_act_pusht_checkpoint_selection_5ep.yaml`。

## 8. 当前因果判断与边界

- ACT并非理论上无法完成PushT；当前模型已学到有效接触、推动和近目标行为。
- offline loss/固定数据MAE改善没有转化为稳定closed-loop success，存在offline-to-online objective mismatch。
- checkpoint selection是实质因素，但不能解释所有失败。
- 缩短replanning、temporal ensemble和早期checkpoint选择都不能单独修复。
- 更可能的剩余因素包括普通BC的闭环分布偏移、数据缺少恢复/终局精调状态、单帧观测和ACT VAE的
  训练—部署latent差异。
- 当前5 seeds足够支持受控诊断，不足以给出正式benchmark置信区间。

## 9. 路线B：最小deterministic ACT消融

用户已认可该方案，但没有授权训练。

主假设：当前ACT默认`use_vae=true`、`kl_weight=10`；训练posterior latent依赖真实action，而部署没有
真实action并使用zero latent。PushT接近单一专家行为，使用`use_vae=false`可能减少训练—部署错位。

第一轮只允许以下核心变化：

```yaml
policy:
  use_vae: false
```

建议预算：

```text
steps: 30,000
save checkpoints: 10k / 20k / 30k
offline eval: every 5k
closed-loop selection: each checkpoint, same five seeds, exec-10
estimated training time: 30-40 minutes
estimated total gate time: 1-1.5 hours
```

必须保持：固定dataset revision和split、batch 8、BF16、chunk/action 50/10、无pretrained backbone、
相同optimizer/seed，以及严格offline/W&B/Hub/recording关闭。

预注册停止标准：

- 任一checkpoint至少1/5 success：形成初步成功demo，再单独决定是否扩大评测；
- 10k/20k/30k全部0/5：立即终止PushT，不追加训练长度、学习率、seed或预训练权重；
- reward改善但0/5只能报告为诊断改善，不能声称成功策略。

第一轮不得同时加入ImageNet pretrained weights、多帧观测、loss weighting、数据增强、exec变化或
temporal ensemble，否则会破坏`use_vae=false`的因果解释。

## 10. 下一对话的严格工作顺序

1. 完整读取本handoff。
2. 只读核对`git status`、HEAD、三个submodule pins。
3. 读取formal 100k配置、formal结果、checkpoint manifest、checkpoint-selection结果与ACT README。
4. 创建路线B 30k训练配置，但不运行训练。
5. 创建严格离线checker，逐字段对照formal配置，只允许`use_vae`、steps、output/job、eval/save cadence
   发生预注册变化；固定全部安全边界。
6. 运行checker并写小型preflight结果/README；不得加载网络资源或启动训练。
7. 向用户报告PASS/FAIL、配置diff、预算、停止标准和唯一宿主训练命令。
8. 等待用户明确授权路线B训练；没有授权不得执行。
9. 用户粘贴训练或评测输出后，必须先判定PASS、FAIL或预期例外，再分析loss/eval/grad/吞吐/显存/时间、
   更新结构化结果与可复现资产，最后给下一条最小安全命令。

下一阶段配置与preflight建议形成独立提交：

```text
experiment: add deterministic ACT ablation config
```

该commit仍需要用户届时另行明确授权；路线B训练结果应再单独提交。
