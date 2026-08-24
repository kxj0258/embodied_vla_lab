# OpenVLA Inference and Fine-tuning

状态：planned；Phase 3 开始。

## 目的

理解 OpenVLA 的图像、语言和动作通路，在 12 GB Blackwell GPU 上完成 4-bit inference，并评估小数据 QLoRA 是否可行。

## 执行边界

- 不从零训练 7B 模型。
- 不运行全参数微调。
- 不把标准 LoRA 设为本机必达目标。
- 一次只下载一个目标 checkpoint。
- batch size 1 起步，先测显存再决定是否训练。
- 第一轮不用 FlashAttention，降低 CUDA extension 风险。

## 计划步骤

1. 阅读 processor、visual encoder、LLM、action tokenizer 和 `predict_action`。
2. 核对 checkpoint action dimension 与 normalization statistics。
3. 4-bit 单图推理。
4. LIBERO 单任务闭环评测。
5. QLoRA forward/backward 显存探测。
6. go/no-go 后再准备小数据微调。

## 必须输出

- image/language/action shape 流程。
- action token 到连续动作的反归一化说明。
- 峰值显存、单步延迟和控制 FPS。
- 量化对动作误差和成功率的影响。
- QLoRA 可行或不可行的证据。
