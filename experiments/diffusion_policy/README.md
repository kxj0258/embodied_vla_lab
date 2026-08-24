# Diffusion Policy

状态：planned。

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
