# Small Action-conditioned World Model

状态：planned。

## 目的

学习动作条件下的短期视觉/latent dynamics，而不是训练大型视频生成模型。

## 第一版模型

```text
image_t → frozen or small encoder → latent_t
latent_t + action_t → latent_{t+1:t+H}
```

优先使用低分辨率、小 horizon 和小数据。首先回答“模型是否使用了 action”，再讨论 imagination 和 planning。

## 基线

- copy-current-frame/latent。
- 无 action dynamics。
- action-conditioned one-step model。
- action-conditioned multi-step rollout。

## 必须输出

- reconstruction/prediction metric。
- one-step 与 multi-step error。
- action ablation。
- 可视化预测。
- 简单候选动作排序实验或明确的不可行结论。
