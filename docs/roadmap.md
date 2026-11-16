# Embodied VLA Lab Roadmap

## 当前进展（2026-08-27）

- [x] Phase 0：`evla-core`、CUDA/BF16、LeRobot、PushT、MuJoCo EGL与LIBERO offline runtime接口。
- [x] Phase 1A：固定PushT dataset revision、206 episodes数据审计与185/21 split。
- [x] Phase 1B：VAE ACT 100k与deterministic ACT 30k训练、checkpoint、闭环及100-seed对照。
- [x] Phase 1C：Diffusion Policy 30k EMA baseline、checkpoint选择、100-seed对照与成功视频。
- [x] PushT受控复现profile、结构化结果和checkpoint/video哈希清单。
- [x] Phase 2A runtime gate：LIBERO offline reset/step与双相机EGL RGB。
- [x] Phase 2A task/data gate：静态、50-demo、40/10 split与预处理合同PASS；接受task replay PASS / legacy state replay FAIL边界。
- [x] Phase 2B ACT训练：资源/overfit/smoke及100k正式训练PASS；完整offline validation与23个checkpoint完整性已冻结。
- [x] Phase 2B ACT环境开发评测：step-100k在固定5个init states上5/5 strict success，端到端适配PASS。
- [x] Phase 2B ACT checkpoint selection：早/中/末三候选后按冻结规则选择step-100k。
- [x] Phase 2B ACT final：未用于开发的init states 5–49完整评测45/45 strict success，Phase 2B完成。
- [x] Phase 3A环境/下载门：独立`evla-openvla`、NF4 `sm_120` kernel及冻结官方LIBERO-Goal 15.085 GB snapshot哈希PASS。
- [x] Phase 3A本地源码导入门：绕过训练专用RLDS/dlimp初始化，四个HF推理类无CUDA导入PASS。
- [x] Phase 3A推理门：4-bit真实单图7D动作PASS；峰值allocated/reserved VRAM约4.71/4.84 GiB，单次冷生成1.024 s。
- [x] Phase 3A环境门静态合同：`evla-core`环境进程与`evla-openvla`模型进程通过哈希绑定IPC隔离，严格单step、超时和清理合同PASS。
- [x] Phase 3A环境单step门：固定init state 0完成1 reset、1 OpenVLA预测和1个被接受的环境step，异常清理与资源记录PASS。
- [x] Phase 3A单episode静态门：persistent worker单次加载、逐步哈希IPC、超时/终止/清理合同PASS，无CUDA执行。
- [x] Phase 3A单episode宿主门：development init state 0在85步strict success，worker/environment清理PASS；N=1不作成功率结论。
- [x] Phase 3A小型development rollout静态门：冻结init states 0–4、单次模型加载、5 episodes与独立success记录合同PASS。
- [x] Phase 3A小型development rollout宿主门：states 0–4为5/5 strict success；Wilson 95%约56.6%–100%，仅开发证据。
- [x] Phase 3A base-policy held-out final静态门：states 5–49、seeds 20260901–20260945及不可回改协议已冻结。
- [x] Phase 3A base-policy held-out final宿主门：states 5–49完整枚举45/45 strict success；Phase 3A base-policy feasibility完成。
- [x] Phase 3B QLoRA完成：工程/离线改善PASS，但heldout 43/45低于base 45/45，环境non-regression FAIL；base保留为可靠demo策略，禁止post-heldout调参。
- [x] P0 MVP收口：OpenVLA base、ACT、Base-vs-QLoRA两类视频、架构图、量化图、manifest、只读核验、portfolio与简历材料完成；ACT展示capture为83步strict success，P0 COMPLETE。
- [x] P0-HD：640真实渲染素材与对比已COMPLETE；OpenVLA base init5 strict success，init40 base/QLoRA为221/70步且均成功，init44 base 68步成功对QLoRA 600步timeout。两个1920×1080并排HD视频已离线合成并hash-manifest；冻结43/45与禁止调参/final重跑边界不变。
- [x] P0-Breadth COMPLETE：10-task/30-episode冻结base评测为20/30 strict success（Wilson 95% 48.78%--80.77%），10/10任务至少成功1次，10个timeout全部保留。固定首成功state的10/10个640 HD recapture逐一复现，共1,296步；1920×1080、64.8秒十任务montage与hash manifests、视觉QA均完成，不改写20/30。
- [x] Phase 4A camera gate：v1相机名错误FAIL资产保留；v2一次reset、0 step/0 prediction运行时PASS。agentview/eye-in-hand的128与640图像、非恒定性、跨视角MAE、hash和视觉QA均PASS。下一步实现并静态冻结Phase 4B development paired rollout，不得从camera gate推断策略success。
- [x] Phase 4B direct-substitution development NO-GO：执行PASS，但agentview 10/15、5/5任务覆盖，eye-in-hand 0/15、0/5任务覆盖；配对为10 agent-only、0 ego-only、5 neither，双侧精确p=0.001953。15个wrist回合全部600步timeout。Final tasks `[1,2,4,5,9]`保持未触碰且不执行；下一步须预注册新的ego-view适配假设，而非直接跑final。
- [~] Phase 4C wrist adaptation：20-update训练工程PASS，checkpoint/optimizer/RNG完整；val10 loss改善32.29%，但token accuracy从16/70降至13/70，混合offline信号不允许checkpoint选择或闭环。下一门为冻结889帧wrist full validation；offline指标不得当作success，final tasks继续未触碰。
- [x] Phase 4D wrist fullval889：配对889帧完成；step-20相对base的loss -45.77%、token accuracy +7.73个百分点、L1 -32.43%，10/10 episode三项均改善；吞吐-13.28%、延迟约+15.5%、显存约+4.7–5.8%。判定offline adaptation PASS、environment recovery unknown，不能当作success/reward。
- [x] Phase 4E development-only wrist rollout gate：执行PASS但策略NO-GO；step-20在task 7 wrist仍为0/3，三个600步timeout/reward 0，与base为3个neither-success配对。offline改善未恢复闭环，冻结失败并停止该checkpoint路线；final tasks保持未触碰。
- [x] Phase 4收口：camera可行、直接替换NO-GO、offline wrist adaptation PASS、闭环恢复NO-GO的完整证据链、项目表述和hash-manifest图已完成。任何后续视角研究必须是新预注册机制，而非继续调step-20。
- [x] P0公开报告与复现索引：完整project report、claim→config/result/manifest/verifier映射、证据语义和fresh-run规则已形成独立文档。
- [x] Phase 5A world-model数据门：40/10 episode时序split、H=1/2/4/8窗口、图像/action/state/reward/done对齐及copy-frame baseline均CPU核验PASS；允许进入模型静态合同，不授权训练。
- [x] Phase 5B world-model静态门：约1.29M参数的action-free/conditioned模型在真实序列上完成H=8 CPU forward/backward，16/16 shape/range/gradient/action-path/resource检查PASS；只允许实现短训练smoke，正式训练仍未授权。
- [x] Phase 5C paired training smoke：工程PASS，两个100-step checkpoint/optimizer完整且loss约降72.5–72.9%；但H4 MAE action-free/conditioned为11.526/11.722，远差于copy-frame 1.732，且conditioned各H均差约1.7–1.8%。offline baseline NO-GO，不选择checkpoint或授权正式训练。
- [x] Phase 5D residual-copy静态门：新模型初始化严格等于copy-frame，32-train/8-development嵌套split与prior-exposed confirmation边界冻结；真实H4 forward/backward及action latent路径12/12 PASS。仅允许实现development smoke。

当前PushT结论：deterministic ACT、VAE ACT、Diffusion EMA的独立100-seed success分别为6%、2%、4%。
Diffusion的配对dense reward与峰值接近度稳定优于ACT，但没有建立更高的稀疏success，且100-step DDPM
推理约为deterministic ACT的39倍。下一阶段不再追加同数据PushT训练，除非预注册新的数据或推理消融。

## 总体原则

项目采用 simulation first、interface first、measurement first 的路线。每个阶段都有明确 gate；上一阶段的数据契约、动作语义或评测闭环未通过时，不增加模型规模。

目标时间为 2026 年 10–11 月，每天可投入约 3–4 小时。路线分为秋招 MVP 和后续增强两部分。

## 秋招 MVP：8 周

| 周次 | 阶段 | 核心任务 | Gate / 可交付产物 |
|---|---|---|---|
| Week 1 | Phase 0 | 驱动、PyTorch、MuJoCo、EGL、项目环境 | CUDA tensor、离屏渲染、环境锁定文档 |
| Week 2 | Phase 1A | 读取 PushT 数据；分析 observation、action、时间窗口和 normalization | 单样本检查报告、ACT 配置说明 |
| Week 3 | Phase 1B | 单 GPU ACT 训练与 rollout | 曲线、checkpoint、视频、成功指标 |
| Week 4 | Phase 1C/2A | Diffusion Policy 对照；LIBERO 单任务接入 | 同数据对照、LIBERO reset/step/video |
| Week 5 | Phase 2B | ACT on one LIBERO-Spatial task | 20–50 rollouts、成功率、FPS、失败分类 |
| Week 6 | Phase 3A | OpenVLA 4-bit pretrained inference | 峰值显存、单步延迟、动作检查、rollout |
| Week 7 | Phase 3B/4 | QLoRA 可行性探测或 ego-view 视角消融 | go/no-go 结论、视角对照结果 |
| Week 8 | MVP 收口 | Demo、README、架构图、简历表述和复现实验检查 | 可展示仓库与一键复现说明 |

## Phase 0：环境准备

目标：建立两个相互隔离的环境，并首先完成 `evla-core`。

- `evla-core`：LeRobot、PushT、ACT、Diffusion Policy、LIBERO、MuJoCo。
- `evla-openvla`：OpenVLA 量化推理和 QLoRA；到 Phase 3 再创建。

Gate：

- PyTorch 识别 RTX 5070 Ti 和 compute capability 12.0。
- CUDA tensor 运算正常。
- MuJoCo EGL 离屏渲染得到有效 RGB 图像。
- LeRobot 和 LIBERO 最小 import/reset/step 正常。
- 精确记录 Python、PyTorch、CUDA runtime、驱动和 commit。

## Phase 1：LeRobot 与 ACT

先使用 PushT，避免同时引入语言、MuJoCo、复杂相机和长时序控制。

学习重点：

- LeRobotDataset v3 目录与 episode 索引。
- 图像/state/action feature key。
- action chunk 与 padding mask。
- normalization、preprocessor 和 postprocessor。
- ACT encoder/decoder、latent variable 和 temporal aggregation。

输出：训练曲线、checkpoint、rollout video、成功指标、峰值显存和失败案例。

## Phase 2：LIBERO Simulation

从 `libero_spatial` 的一个短任务开始，再扩展任务数量。

必须确认：

- 相机图像名称、分辨率和裁剪。
- 机器人 proprioception。
- 7D action 的坐标系、旋转表示、gripper 语义和控制频率。
- episode termination 与 success condition。
- demonstration 与 evaluation 环境的一致性。

输出：成功率、推理 FPS、控制频率、失败分类和视频。

Phase 2B前置门保留完整历史：400-step单批过拟合PASS；full-split smoke v1因checkpoint序列化FAIL；
Codex管理环境v2因CUDA不可见记为EXPECTED_EXCEPTION；宿主v3及validation smoke v4均PASS。

100k正式训练随后PASS：step-0到100k共101次完整889-frame validation，最低loss 0.426634位于step 1,000，
final为0.440030；train loss继续下降而validation长期平台，不能据此推断后期checkpoint的环境控制更好。
22个4,460-step精确恢复边界和一个100k archival checkpoint均已哈希并CPU重载。下一门是独立、全新ID的
LIBERO checkpoint rollout；在该门完成前success、reward、rollout FPS和策略checkpoint排名均未知。

step-100k首次独立development rollout随后在init states 0–4上取得5/5 strict success，所有episode reward均为
1.0，平均80.6环境步、3.49秒/episode；纯rollout吞吐约23.09 env steps/s。该结果验证图像旋转、9D state、
动作反归一化、ACT执行窗口及LIBERO success判定的端到端闭环，但N=5的Wilson 95%区间约56.6%–100%，
不能称为正式100%成功率。下一步用完全相同的5-state协议评测最早周期checkpoint step 4,460，判断训练
预算是否改善闭环，再冻结一次未用于选择的held-out final协议（后续修正为剩余45 states）。

同协议step-4,460评测执行PASS，策略为4/5 success：states 0--3成功，state 4在600步超时；step-100k在
相同state 4上78步成功，因此当前配对开发结果以5/5对4/5支持继续训练带来闭环收益。只有一个discordant
pair，不能称统计显著。为限制checkpoint挑选偏差，只再运行一个预先固定的中点step-49,060候选，然后
按success、reward、完成步数的层级规则冻结checkpoint并进入未用于选择的held-out final。

中点step-49,060同样取得5/5、mean reward 1.0；其总完成步数407，step-100k为403。按冻结规则最终选择
step-100k。4步差异只是开发集tie-break，不代表显著优越。由于任务总计只有50个init states且0–4已经
用于开发，不存在“未使用的50-state final”；正式协议修正并冻结为剩余indices 5–49的45-state held-out
final。该final不能用于回改checkpoint选择。

冻结的step-100k随后在完整held-out finite set（init states 5–49）取得45/45 strict success，所有episode
sum/max reward均为1.0；平均/中位完成步数79.64/79，范围74–106，纯rollout约1.81秒/episode。若将episode
外推为Bernoulli总体，Wilson 95%约92.1%–100%；对冻结的45-state有限集合则是完整枚举而非抽样。
Phase 2B至此完成。该结论不覆盖其他LIBERO任务、训练seed、相机扰动、语言泛化或不同simulator版本。

## Phase 3：OpenVLA

Phase 3A static/environment/download gates and the NF4 single-image gate PASSed on 2026-08-27. The official
suite-finetuned `openvla/openvla-7b-finetuned-libero-goal` revision `fa5ae1e...9baf` is locally hash-frozen.
One real LIBERO image produced a finite 7D action in 1.024 s cold inference. The dependency-isolated single-step
bridge now also PASSes static schema, hash, subprocess and cleanup checks without constructing an environment or
loading the model. The formal one-step host gate then PASSed: one online image produced a finite in-range action,
LIBERO accepted exactly one step, reward was 0, no termination occurred, and the environment closed safely. This is
interface evidence, not policy success evidence. Next: a separately implemented persistent-worker single episode,
then a small development set only if that episode gate is stable.

顺序：

1. 阅读 processor、visual encoder、LLM、action tokenizer 和 `predict_action`。
2. 只下载一个与目标 suite 对应的 checkpoint。
3. 4-bit 单图推理。
4. 4-bit LIBERO 闭环评测。
5. batch size 1 的 QLoRA 显存探测。
6. 通过 go/no-go 后才进行小数据微调。

普通 LoRA 的显存要求超过本机预算，因此不把它设为本机必达指标。QLoRA 也必须经过峰值显存和数值稳定性验证。

## Phase 4：Ego-view Extension

研究变量：相机外参、遮挡、尺度、运动模糊、图像增强、动作坐标和 normalization。

优先做受控对照：固定任务、数据量、策略和训练步数，只改变 camera setup。

## Phase 5：Small World Model

不训练大规模视频生成模型。优先选择：

```text
image_t → frozen/small encoder → latent_t
latent_t + action_t → latent_{t+1:t+H}
```

比较复制当前帧、无动作模型和 action-conditioned 模型，评估单步与多步误差，以及是否能支持简单候选动作排序。

当前状态：Phase 5A数据合同与Phase 5B小模型静态门PASS。Phase 5C完整帧生成模型H4 MAE为11.526/11.722，远差于copy-frame 1.732，冻结为offline baseline NO-GO。Phase 5D的copy-preserving bounded residual机制静态PASS；Phase 5E在隔离32/8 development split上执行PASS，但H4为copy 1.59125、free 1.59223、conditioned 1.59303，conditioned未击败任何基线，结论为development NO-GO。原10个prior-exposed confirmation episodes未执行；禁止确认集、checkpoint选择、加长训练或环境/planning主张。Phase 5停止当前机制，下一主线转向Phase 6 Feagine bridge的CPU静态接口与仿真前安全合同。

## Phase 6：Feagine Transfer

建议接口：

```text
image + instruction
        ↓
VLA high-level action / target
        ↓
Feagine action adapter
        ↓
tip target + constraints
        ↓
PCC IK/controller
        ↓
MuJoCo continuum robot
```

第一版任务：Move to target。随后增加 obstacle avoidance 和 contact interaction。

Phase 6A状态：通用7D delta→tip target→单段可伸缩PCC参考接口已实现，256次往返最大位置误差8.58e-16 m，14/14静态与失败安全检查PASS。Phase 6B随后从v1.0.1发行包绑定实际A03 Type-2合同：3×0.16m固定段、每段5采样、6个θ/φ变量、48个proxy关节/actuator、两指及旋转stage、5ms timestep，静态19/19 PASS。包默认`a03`但实际目录仅`a03_type_2`，调用必须显式preset。下一门是在新隔离环境安装core/wheel并做0-step MJCF/runtime gate；外部policy-to-A03 frame仍待标定。

Phase 6C v1已完成隔离环境安装、core编译和wheel安装。zero-step runtime为10/11 FAIL，仅因静态审计把`<default><joint>`模板计入52，MuJoCo实际正确实例化51 named joints/51 actuators；其余preset、控制绑定、FK/IK和data.time=0全部PASS。修正计数的v2静态9/9 PASS，将复用环境、不重装不编译。v2通过前不授权simulation step。

Phase 6C v2现为11/11 PASS：core IK 5次迭代，位置/方向误差74.67 μm/1.35e-4，data.time严格0→0。Phase 6D预注册200 steps×5ms的一秒headless actuator/dynamics smoke，检查有限状态、joint limits、tip motion和精确时间；它不包含IK target、reward或success。

Phase 6D v1执行6/7 FAIL：200步/1秒、有限状态、tip motion 0.0670m、最大qvel 0.592、0 contact均正常，但joint-limit检查失败。v1 FAIL永久保留。相同命令/步数的v2逐关节诊断执行PASS：仅`feagine_grasper_joint_1`超过slide上限11.77 μm，为预注册0.1 mm材料阈值的11.77%，无hinge越界，判定`SOFT_LIMIT_PENETRATION_WITHIN_DIAGNOSTIC_THRESHOLD`。该结果只解释v1机制，不自动授权target-reaching；下一步须单独冻结低幅、无渲染的target-reaching smoke合同。

Phase 6E bounded move-to-target执行9/9 PASS。单个core-generated目标初始距离53.90 mm，400步/2秒后误差9.50 mm、下降82.37%；IK 4次迭代，状态/时间正常、无材料限位越界、0 contact。该门不使用VLA、图像、reward、障碍物或硬件；PASS只允许下一步冻结外部指定的多目标development协议。

Phase 6F外部显式目标development结果为0/4 FAIL。±X的core IK在1次迭代后失败且末端不动；±Y的IK在16次迭代数值成功，但MJCF最终误差由35 mm扩大到241.63/271.86 mm。所有状态/时间正常、无材料越界、0 contact，因此下一步只能做保持四目标不变的映射诊断，记录solved、clamped/applied section angles、actuator targets与core-vs-MJCF位移；不得换目标、放宽门槛或进入VLA。

Phase 6G映射诊断执行PASS并定位两层根因：±X从全零直杆初值在IK第1次迭代停滞；±Y raw解的负θ被A03 preset clamp为0（+Y一段、-Y两段），破坏等价构型。clamp后的core预测与MJCF实际cosine>0.999，排除主要actuator/dynamics方向错误。下一门应静态证明`θ<0→|θ|, φ→wrap(φ+π)`保持FK等价，并冻结对称multi-start seed；随后才可用原四目标新ID做恢复门。

Phase 6H CPU-only结果为4/5 FAIL。signed-θ规范化对256随机+4实测解保持FK等价（最大位置9.22e-8 m、方向4.28e-7），但8组对称multi-start对四目标均无法达到0.1 mm容差，最佳残差集中1.592–1.615 mm。说明Phase 6F“最大伸展高度不变+35 mm横移+方向不变”的目标合同本身疑似不可达。下一门必须先量化固定方向可达域与所需轴向回撤，不能放宽容差或进入MJCF恢复。

Phase 6I可达域扫描完成：保持35 mm横移、方向和0.1 mm容差，0–5 mm按0.25 mm扫描；四方向均在1.50 mm回撤首次可达，spread=0，1.25 mm仍约0.347 mm误差、1.50 mm约0.0982 mm。v1唯一失败是浮点精确网格比较，v2以1e-12校验同一证据10/10 PASS且0次新IK。下一步可冻结35 mm横移+1.50 mm回撤的Phase 6J MJCF recovery dev4，但必须保留Phase 6F 0/4且使用signed-θ规范化和固定seed选择。

Phase 6J recovery dev4执行2/4 FAIL。四方向均通过IK、≤15 mm终端误差和≥70%进度（约9.30 mm/73.44%），证明1.50 mm回撤+signed-θ修复了目标运动；±Y完整PASS，±X只因slide穿透约0.2631 mm超过0.1 mm材料阈值失败。v1未记录关节名/时刻，下一步只能做同目标逐关节限位诊断；不得放宽阈值、删除±X或进入VLA。

Phase 6K逐slide诊断执行PASS：+X为`feagine_grasper_joint_0`、−X为镜像`joint_1`，均在step 1由−13 mm硬下限瞬态穿透至约−13.2631 mm；对侧指仅约2.45 μm，±Y≤3.90e-11 m。下一步可预注册gripper command小裕量扫描，选择离开硬下限且仍保持目标误差/进度的最小值；不得放宽0.1 mm阈值或修改臂目标。

Phase 6L执行本身PASS，但选型结论为`NO_SAFE_GRIPPER_MARGIN_SELECTED`：0.00/0.01/0.02/0.05均只有2/4，±X穿透随command单调由约0.2631降至0.1150 mm，±Y始终PASS，臂末端误差/进度保持约9.30 mm/73.44%。最大预注册command 0.05仍比冻结0.1 mm阈值高约0.015 mm，因此不得选择margin或把结果写成恢复成功。若继续，必须建立全新、一次性、预注册的确认门；Phase 6L和Phase 6J结论不得回改。

## 11 月增强目标

- 完成小型 world model。
- 建立 Feagine simulation bridge。
- 完成至少一个语言指令到连续体控制闭环 Demo。
- 将关键实验整理成项目报告、演示视频和简历材料。
