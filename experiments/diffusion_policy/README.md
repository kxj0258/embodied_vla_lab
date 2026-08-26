# Diffusion Policy

状态：Phase 1C baseline 30k训练执行PASS；checkpoint闭环选择待执行。

## 目的

在与 ACT 相同的数据、observation、action 和评测协议下建立 Diffusion Policy 对照，理解生成式连续动作建模与 action chunk 推理成本。

## 对照原则

- 使用与 ACT 相同的数据 split 和 seed。
- 保持相同图像输入和 action representation。
- 同时报告成功率、训练时间、推理延迟和峰值显存。
- 不用单一 training loss 直接判断策略优劣。

## 必须输出

- 模型与 noise scheduler 说明。
- denoising steps 与延迟关系。
- ACT/DP 定量对照表。
- failure case 对照。

## Phase 1C：同数据30k baseline

首个候选固定使用与deterministic ACT 30k完全相同的PushT revision、185/21 episode split、batch 8、
BF16、seed、无预训练backbone、每5k离线评测及10k/20k/30k checkpoints。实际执行窗口仍为10步。

Diffusion实现要求prediction horizon能被三层U-Net的下采样因子8整除，因此使用最接近ACT chunk 50的
`horizon=48`；同时采用两帧观测、DDPM 100 train/inference steps、模型原生MIN_MAX state/action
normalization、Adam+cosine warmup和EMA。这是强模型族baseline，不是只改变policy type的单字段消融；
全部不可匹配项已在`results/phase1c_diffusion_pusht_baseline_30k_preflight.yaml`预注册。

训练配置为`configs/train/diffusion_pusht_phase1c_baseline_30k.yaml`，严格离线checker为
`scripts/check_phase1c_diffusion_baseline_config.py`。任何安装、训练和闭环评测仍需分别明确授权。

首次严格离线preflight完成静态配置与dataset contract检查：185/21 episodes、23043/2607 frames均
精确匹配，未创建output目录、未初始化CUDA、未改变dataset payload。经单独授权，`evla-core`安装了
项目约束和lockfile一致的`diffusers==0.39.0`；重跑依赖gate及完整preflight退出状态为0，判定PASS。
TorchCodec ABI warning继续按既有协议回退到显式PyAV，不构成阻塞。短GPU资源gate随后经单独授权执行。

### GPU资源gate PASS

经单独授权完成2-step、batch-8 BF16资源测试，trainer和`time -v`退出状态均为0，无OOM、NaN/Inf或
socket通信。模型共262,709,026个参数，EMA启用。Step 1/2 loss为1.047/1.257，pre-clip grad norm为
20.137/23.006；两步都处于500-step warmup最初阶段且样本极少，只证明数值有限，不能解释收敛趋势。

峰值PyTorch allocation为5.94 GiB，约占RTX 5070 Ti Laptop 12,227 MiB总显存的49%，通过低于90%的
资源停止线。首步含初始化耗时1.079秒，第二步0.227秒；用第二步外推纯30k更新约1.89小时，计入6次完整
validation、checkpoint I/O和热降频后保守预计2–3小时，仍低于4小时停止线。资源测试未保存checkpoint，
甚至没有创建output目录。完整记录见`results/phase1c_diffusion_pusht_resource_gate_2step.yaml`。

下一步是正式30k训练，仍需新的明确授权；资源gate授权不自动扩展到正式训练。

### 正式30k训练执行PASS / success pending

宿主严格离线完成30,000个batch-8 BF16 optimizer steps，进程exit status为0；六次完整validation、
10k/20k/30k checkpoints均齐全，每个checkpoint同时保存live和EMA权重，`last`指向030000。未发现
NaN/Inf、OOM、swap或socket通信。训练执行判PASS，但没有执行环境rollout，success仍为PENDING。

150个200-step训练窗口中，首/末5窗口loss均值从0.1468降至0.0218，下降85.15%；pre-clip grad norm
均值从3.4758降至0.3606，全部窗口均低于clip阈值10。峰值PyTorch allocation为5.94 GiB，平均吞吐
45.59 samples/s。全进程wall time为1:32:15，明显短于2–3小时保守估计；maximum RSS 4,489,808 KiB
是主机内存，不是GPU显存。

live-weight offline eval依次为`0.0527/0.0481/0.0491/0.0557/0.0726/0.0761`，10k最低；最低点到30k
上升58.21%，提示后期过拟合或校准漂移。但trainer的offline eval明确使用live而非EMA权重，不能由此
直接淘汰20k/30k EMA，也不能推断闭环成功。Diffusion loss预测归一化噪声，不能与ACT的动作L1数值比较。

相较deterministic ACT 30k，Diffusion吞吐低68.42%、wall time约3.16倍、峰值PyTorch allocation约
9.58倍；这是263M参数Diffusion模型及EMA带来的资源代价，不是任务效果结论。完整训练结果与六个模型
哈希见`results/phase1c_diffusion_pusht_baseline_30k.yaml`和checkpoint manifest。

下一步按预注册顺序从10k EMA开始，在固定5个development seeds上做同步单环境、300-step、exec-10
闭环选择；在得到checkpoint选择前不运行100个held-out seeds，也不调整denoising steps。

### 10k EMA闭环执行PASS / success FAIL

10k EMA完成固定seeds `20260825..20260829`的五条同步exec-10轨迹，eval与wrapper退出状态均为0，
五个视频和`eval_info.json`齐全、socket为0，因此执行PASS；success为0/5，checkpoint-level gate FAIL。

平均sum/max reward为73.5272/0.58109。逐seed max reward为
`0.54380/0.53048/0.80232/0.36615/0.66268`；最佳seed 20260827的0.80232约对应76.22% peak coverage，
未进入严格near-success区间。相较deterministic ACT 10k，平均sum reward低14.26%，max reward高12.52%，
说明Diffusion在部分状态更接近目标，但尚未转化为成功。

100-step DDPM纯eval耗时250.91秒，即50.18秒/episode，约为deterministic ACT 10k的20.55倍；全进程
wall time 4:18.86。Maximum RSS 3,386,416 KiB是主机内存，本输出未报告GPU allocation。训练loss、
offline eval、grad norm、训练吞吐和训练显存没有因本次rollout改变。

当前只完成10k EMA，不能触发全模型FAIL，也不能提前调低denoising steps。下一步保持完全相同协议只评测
20k EMA；100-seed held-out集合继续冻结不用。

### 20k EMA闭环执行PASS / success FAIL

20k EMA完成相同五seed协议，eval与wrapper exit status均为0，五个视频、eval_info与raw log齐全，
socket为0，因此执行PASS；success仍为0/5，checkpoint-level gate FAIL。

平均sum/max reward为99.0972/0.60447，相较10k EMA分别提高34.78%/4.02%。逐seed max reward为
`0.59492/0.54707/0.82442/0.43017/0.62577`，最佳仍是seed 20260827，但约78.32% peak coverage仍明显
低于成功阈值。相较deterministic ACT 20k，sum/max reward分别高21.44%/79.37%，但两者均0/5成功，
只能说明20k Diffusion更常靠近目标，不能称为任务成功。

纯eval为248.96秒、49.79秒/episode，与10k EMA基本一致；约为deterministic ACT 20k的19.45倍。
全进程wall time 4:17.21，Maximum RSS 3,384,744 KiB不是GPU显存。训练指标未被此次rollout改变。

10k和20k EMA累计0/10 success，但预注册协议仍要求完成30k EMA后判定。下一步只评测30k EMA，不改变
100次denoising、执行窗口、seeds或其他推理设置。

### 30k EMA闭环执行PASS / development success gate FAIL

30k EMA完成相同五seed协议，执行与资产完整性PASS，但success仍为0/5。因此10k/20k/30k EMA合计
0/15，预注册development success gate整体FAIL，当前不能声称Diffusion已经产生成功demo。

30k平均sum/max reward为128.4161/0.85863，相较20k EMA分别提高29.59%/42.05%。逐seed max reward为
`0.92095/0.92964/0.68573/0.79365/0.96316`，其中3/5超过0.90、1/5超过0.95，但均未满足环境严格
success。这是明确的终局跨阈值或成功保持缺口，而不是完全不会推动。

按预注册选择规则，三个checkpoint success同为0后，以平均max reward、再以sum reward排序，冻结30k EMA。
它相较deterministic ACT 30k的平均sum/max reward高0.75%/15.97%，但success为0/5而ACT为1/5，不能
称为成功率改善。另一方面，live offline eval认为10k最佳，而EMA闭环指标随训练单调改善，再次证明不能
用live offline loss选择EMA部署checkpoint。

30k纯eval为226.86秒、45.37秒/episode，仍约为deterministic ACT的17.62倍；全进程wall time 3:54.08。
若继续既定独立评测，必须使用已经冻结的30k EMA和未接触的`20260830..20260929`共100 seeds，不得根据
结果切换checkpoint或调整100-step denoising。

### 冻结30k EMA独立100-seed评测PASS / 4% success

冻结的30k EMA在与ACT完全相同的100个held-out seeds上执行PASS，获得4/100 success；成功seeds为
`20260860/20260901/20260924/20260929`，Clopper-Pearson exact 95%区间约`[1.10%,9.93%]`。
这证明Diffusion确实能产生成功轨迹，但可靠性仍低，且不能追溯改变开发集0/5 gate的FAIL结论。

平均/中位sum reward为86.8430/83.2745，平均/中位max reward为0.65133/0.69303；3条zero-sum，20条
达到max reward至少0.90，5条至少0.95。相较deterministic ACT 30k，Diffusion success点估计低2pp
（4%对6%），配对McNemar `p=0.7266`、success差bootstrap区间`[-8pp,+3pp]`，没有稳定差异。

Dense指标则稳定改善：相较deterministic ACT，配对平均sum reward高14.489，95% bootstrap区间
`[2.713,26.404]`；平均max reward高0.13576，区间`[0.06350,0.20880]`，均不跨0。相较VAE ACT 25k，
对应差异为+19.520与+0.23907，区间同样不跨0。结论是Diffusion更稳定地推动并接近目标，但没有更稳定
地完成严格终局成功。

100-step DDPM纯eval为4555.23秒、45.55秒/episode，约为deterministic ACT的39.0倍；全进程wall time
1:16:02。评测器只渲染前10条，而四个成功均在其后，因此当前尚无Diffusion成功视频资产。最快获得
可展示demo的下一步是只重放并渲染首个成功seed 20260860，不调整模型或推理参数。

### 单seed重放：预期随机策略例外

单独重放环境seed 20260860执行PASS，但得到0/1 success、max reward 0.78954，没有复现100-seed运行中的
成功。这不是原结果失效：Diffusion Policy每个action chunk还会从Torch RNG采样噪声；原运行中该seed是
episode index 30，policy RNG已被前30集推进，而单集命令从初始policy RNG开始。环境seed相同并不足以
唯一确定随机策略轨迹。

为精确复现，新增`scripts/eval_render_all_episodes.py`包装器：从原始起始seed 20260830顺序运行31集，
保持与100-seed实验相同的环境及policy RNG轨迹，并渲染全部31集；目标成功视频应为episode 30。该方案
不改变模型、denoising或动作协议，但预计约24分钟，且不应把失败的单seed视频当作demo。

### 31集顺序重放：执行PASS / 精确复现FAIL / demo PASS

顺序运行seeds `20260830..20260860`并渲染全部31集，进程exit status为0；得到2/31 success，成功的是
episode 3（seed 20260833）与episode 8（seed 20260838），两者max reward均为1.0。项目因此已经拥有
两个真实Diffusion成功视频，demo资产目标PASS。

原目标episode 30（seed 20260860）本次max reward为0.94397、未成功；重放前10个视频与原100-seed
运行前10个视频的SHA-256全部不同。因此当前CUDA推理/执行栈即使保持环境seed序列和名义Torch RNG
初始化，也不具备跨进程逐轨迹bitwise确定性。精确轨迹复现判FAIL，不能声称episode 30复现成功。

这不推翻原4/100统计：该统计是一次完整、冻结协议的有效采样。新增episode 3/8同样是冻结checkpoint、
未改推理协议下的新成功样本，可作为demo，但必须标注为“顺序重放中新产生的成功”，而不是原成功轨迹。
31集纯eval 1337.99秒、43.16秒/episode，全进程wall time 22:24.79；没有产生新训练指标。
