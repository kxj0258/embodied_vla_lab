# Unified VLA Benchmark Contract

本契约是 Phase 1/2 的统一入口，不替换历史 phase 脚本。历史结果通过
`scripts/benchmark.py import` 进入同一个摘要格式；新的 runtime evaluator
应实现 `evla_benchmark.adapters.PolicyAdapter` 和 `EnvironmentAdapter`。

## Canonical observation

- RGB image: `uint8`, `HWC`, 默认 `128×128×3`；相机名称必须显式记录。
- state: 一维 `float32` proprioception，LIBERO 当前为 9 维。
- instruction: OpenVLA 必须提供，ACT 可为空。
- camera frame、state frame 和图像预处理必须写入结果 provenance。

## Canonical action

- LIBERO 当前为 7D relative end-effector action。
- rotation representation、gripper semantics、coordinate frame 和 normalization 不得隐含。
- `chunk_size` 与 `action_steps_per_inference` 必须同时记录。
- evaluator 必须检查维度、有限值和动作边界，然后才调用环境。

## Result and failure taxonomy

每个 episode 至少包含 `init_state_index`、`seed`、`success`、`steps` 和失败状态。
统一失败类别为：`success`、`timeout`、`termination_without_success`、
`invalid_action`、`inference_timeout`、`numerical_error`、`environment_error`、
`protocol_error`、`unknown`。

## Usage

```bash
python scripts/benchmark.py import \
  results/phase2b_act_libero_ckpt100000_final_heldout45.yaml

python scripts/benchmark.py compare --result \
  results/phase2b_act_libero_ckpt100000_final_heldout45.yaml \
  results/phase3a_openvla_libero_final_heldout45_v1.yaml
```

该入口默认只读，不重跑实验、不下载模型、不覆盖结果目录。
