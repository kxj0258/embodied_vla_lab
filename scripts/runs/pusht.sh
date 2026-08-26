#!/usr/bin/env bash
# Reproducible, strict-offline PushT run profiles.
# This script is inert until a profile is explicitly selected. Every mutating
# profile refuses to overwrite an existing log or output directory.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: scripts/runs/pusht.sh <profile>

Set EVLA_DRY_RUN=1 to print the resolved command without running it.

Audit / preflight:
  act-deterministic-preflight    diffusion-preflight
  act-formal-accuracy            act-deterministic-accuracy

Training (requires separate authorization):
  act-smoke-train                act-overfit-train
  act-resource-gate-train        act-formal-100k-train
  act-deterministic-30k-train    diffusion-resource-gate-train
  diffusion-30k-train

Closed-loop evaluation:
  act-vae-25k-5ep                act-vae-50k-5ep
  act-vae-75k-5ep                act-vae-100k-5ep
  act-vae-100k-exec1-5ep         act-vae-100k-exec5-5ep
  act-vae-100k-temporal-5ep
  act-deterministic-10k-5ep      act-deterministic-20k-5ep
  act-deterministic-30k-5ep      act-deterministic-30k-100seed
  act-vae-25k-100seed
  diffusion-ema-10k-5ep          diffusion-ema-20k-5ep
  diffusion-ema-30k-5ep          diffusion-ema-30k-100seed
  diffusion-demo-replay31
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

PROFILE="$1"
if [[ "$PROFILE" == "--list" || "$PROFILE" == "-h" || "$PROFILE" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "evla-core" ]]; then
  echo "Refusing to run: activate evla-core first." >&2
  exit 2
fi

OFFLINE_ENV=(
  env -u ALL_PROXY -u all_proxy
  HF_HUB_OFFLINE=1
  HF_DATASETS_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  HF_HUB_DISABLE_TELEMETRY=1
  WANDB_MODE=disabled
  CUDA_VISIBLE_DEVICES=0
  PYTHONUNBUFFERED=1
)

LOG_PATH=""
OUTPUT_PATH=""
COMMAND=()

set_train() {
  local config="$1" slug="$2" output="$3"
  LOG_PATH="results/raw/${slug}.log"
  OUTPUT_PATH="$output"
  COMMAND=(lerobot-train "--config_path=${config}")
}

set_eval() {
  local checkpoint="$1" slug="$2" episodes="$3" seed="$4"
  shift 4
  LOG_PATH="results/raw/${slug}.log"
  OUTPUT_PATH="outputs/eval/${slug}"
  COMMAND=(
    lerobot-eval
    "--policy.path=${checkpoint}"
    --policy.device=cuda
    --policy.use_amp=false
    --env.type=pusht
    --env.task=PushT-v0
    --env.fps=10
    --env.episode_length=300
    --env.obs_type=pixels_agent_pos
    --env.render_mode=rgb_array
    --env.observation_height=96
    --env.observation_width=96
    --env.visualization_height=384
    --env.visualization_width=384
    "--eval.n_episodes=${episodes}"
    --eval.batch_size=1
    --eval.use_async_envs=false
    --eval.recording=false
    "--seed=${seed}"
    "--job_name=${slug}"
    "--output_dir=${OUTPUT_PATH}"
    --trust_remote_code=false
    "$@"
  )
}

case "$PROFILE" in
  act-deterministic-preflight)
    LOG_PATH="results/raw/phase1b_act_pusht_deterministic_30k_preflight_rerun.log"
    COMMAND=(python scripts/check_phase1b_act_deterministic_30k_config.py)
    ;;
  diffusion-preflight)
    LOG_PATH="results/raw/phase1c_diffusion_pusht_baseline_30k_preflight_rerun.log"
    COMMAND=(python scripts/check_phase1c_diffusion_baseline_config.py)
    ;;
  act-formal-accuracy)
    LOG_PATH="results/raw/phase1b_act_pusht_formal_100k_checkpoint_accuracy_rerun.log"
    COMMAND=(python scripts/check_phase1b_act_checkpoint_accuracy.py
      --checkpoint outputs/train/phase1b_act_pusht_formal_100k_bf16/checkpoints/100000/pretrained_model
      --dataset-contract configs/dataset/pusht_v3.yaml --batch-size 8)
    ;;
  act-deterministic-accuracy)
    LOG_PATH="results/raw/phase1b_act_pusht_deterministic_30k_checkpoint_accuracy_rerun.log"
    COMMAND=(python scripts/check_phase1b_act_checkpoint_accuracy.py
      --checkpoint outputs/train/phase1b_act_pusht_deterministic_30k_bf16/checkpoints/030000/pretrained_model
      --dataset-contract configs/dataset/pusht_v3.yaml --batch-size 8)
    ;;
  act-smoke-train)
    set_train configs/train/act_pusht_phase1b_smoke.yaml phase1b_act_pusht_smoke_2step outputs/train/phase1b_act_pusht_smoke_bf16
    ;;
  act-overfit-train)
    set_train configs/train/act_pusht_phase1b_overfit_ep51.yaml phase1b_act_pusht_overfit_ep51_2000step outputs/train/phase1b_act_pusht_overfit_ep51_bf16
    ;;
  act-resource-gate-train)
    set_train configs/train/act_pusht_phase1b_fullsplit_batch8_100step.yaml phase1b_act_pusht_fullsplit_batch8_100step outputs/train/phase1b_act_pusht_fullsplit_batch8_100step_bf16
    ;;
  act-formal-100k-train)
    set_train configs/train/act_pusht_phase1b_formal_100k.yaml phase1b_act_pusht_formal_100k outputs/train/phase1b_act_pusht_formal_100k_bf16
    ;;
  act-deterministic-30k-train)
    set_train configs/train/act_pusht_phase1b_deterministic_30k.yaml phase1b_act_pusht_deterministic_30k outputs/train/phase1b_act_pusht_deterministic_30k_bf16
    ;;
  diffusion-resource-gate-train)
    set_train configs/train/diffusion_pusht_phase1c_resource_gate_2step.yaml phase1c_diffusion_pusht_resource_gate_2step outputs/train/phase1c_diffusion_pusht_resource_gate_2step_bf16
    ;;
  diffusion-30k-train)
    set_train configs/train/diffusion_pusht_phase1c_baseline_30k.yaml phase1c_diffusion_pusht_baseline_30k outputs/train/phase1c_diffusion_pusht_baseline_30k_bf16
    ;;
  act-vae-25k-5ep)
    set_eval outputs/train/phase1b_act_pusht_formal_100k_bf16/checkpoints/025000/pretrained_model phase1b_act_pusht_checkpoint_025000_5ep_exec10 5 20260825 --policy.n_action_steps=10
    ;;
  act-vae-50k-5ep)
    set_eval outputs/train/phase1b_act_pusht_formal_100k_bf16/checkpoints/050000/pretrained_model phase1b_act_pusht_checkpoint_050000_5ep_exec10 5 20260825 --policy.n_action_steps=10
    ;;
  act-vae-75k-5ep)
    set_eval outputs/train/phase1b_act_pusht_formal_100k_bf16/checkpoints/075000/pretrained_model phase1b_act_pusht_checkpoint_075000_5ep_exec10 5 20260825 --policy.n_action_steps=10
    ;;
  act-vae-100k-5ep)
    set_eval outputs/train/phase1b_act_pusht_formal_100k_bf16/checkpoints/100000/pretrained_model phase1b_act_pusht_demo_5ep 5 20260825
    ;;
  act-vae-100k-exec1-5ep)
    set_eval outputs/train/phase1b_act_pusht_formal_100k_bf16/checkpoints/100000/pretrained_model phase1b_act_pusht_demo_5ep_exec1 5 20260825 --policy.n_action_steps=1
    ;;
  act-vae-100k-exec5-5ep)
    set_eval outputs/train/phase1b_act_pusht_formal_100k_bf16/checkpoints/100000/pretrained_model phase1b_act_pusht_demo_5ep_exec5 5 20260825 --policy.n_action_steps=5
    ;;
  act-vae-100k-temporal-5ep)
    set_eval outputs/train/phase1b_act_pusht_formal_100k_bf16/checkpoints/100000/pretrained_model phase1b_act_pusht_demo_5ep_temporal_ensemble 5 20260825 --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01
    ;;
  act-deterministic-10k-5ep)
    set_eval outputs/train/phase1b_act_pusht_deterministic_30k_bf16/checkpoints/010000/pretrained_model phase1b_act_pusht_deterministic_010000_5ep_exec10 5 20260825
    ;;
  act-deterministic-20k-5ep)
    set_eval outputs/train/phase1b_act_pusht_deterministic_30k_bf16/checkpoints/020000/pretrained_model phase1b_act_pusht_deterministic_020000_5ep_exec10 5 20260825
    ;;
  act-deterministic-30k-5ep)
    set_eval outputs/train/phase1b_act_pusht_deterministic_30k_bf16/checkpoints/030000/pretrained_model phase1b_act_pusht_deterministic_030000_5ep_exec10 5 20260825
    ;;
  act-deterministic-30k-100seed)
    set_eval outputs/train/phase1b_act_pusht_deterministic_30k_bf16/checkpoints/030000/pretrained_model phase1b_act_pusht_deterministic_030000_100seed_exec10 100 20260830
    ;;
  act-vae-25k-100seed)
    set_eval outputs/train/phase1b_act_pusht_formal_100k_bf16/checkpoints/025000/pretrained_model phase1b_act_pusht_vae_025000_100seed_exec10 100 20260830
    ;;
  diffusion-ema-10k-5ep)
    set_eval outputs/train/phase1c_diffusion_pusht_baseline_30k_bf16/checkpoints/010000/pretrained_model_ema phase1c_diffusion_pusht_ema_010000_5ep_exec10 5 20260825
    ;;
  diffusion-ema-20k-5ep)
    set_eval outputs/train/phase1c_diffusion_pusht_baseline_30k_bf16/checkpoints/020000/pretrained_model_ema phase1c_diffusion_pusht_ema_020000_5ep_exec10 5 20260825
    ;;
  diffusion-ema-30k-5ep)
    set_eval outputs/train/phase1c_diffusion_pusht_baseline_30k_bf16/checkpoints/030000/pretrained_model_ema phase1c_diffusion_pusht_ema_030000_5ep_exec10 5 20260825
    ;;
  diffusion-ema-30k-100seed)
    set_eval outputs/train/phase1c_diffusion_pusht_baseline_30k_bf16/checkpoints/030000/pretrained_model_ema phase1c_diffusion_pusht_ema_030000_100seed_exec10 100 20260830
    ;;
  diffusion-demo-replay31)
    set_eval outputs/train/phase1c_diffusion_pusht_baseline_30k_bf16/checkpoints/030000/pretrained_model_ema phase1c_diffusion_pusht_ema_030000_replay_through_seed20260860 31 20260830
    COMMAND[0]=python
    COMMAND=(python scripts/eval_render_all_episodes.py "${COMMAND[@]:1}")
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    usage >&2
    exit 2
    ;;
esac

printf 'Profile: %s\n' "$PROFILE"
printf 'Working directory: %s\n' "$ROOT_DIR"
printf 'Command: '
printf '%q ' "${OFFLINE_ENV[@]}" "${COMMAND[@]}"
printf '\nLog: %s\n' "$LOG_PATH"
[[ -n "$OUTPUT_PATH" ]] && printf 'Output: %s\n' "$OUTPUT_PATH"

if [[ "${EVLA_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

[[ ! -e "$LOG_PATH" ]] || { echo "Refusing to overwrite log: $LOG_PATH" >&2; exit 3; }
if [[ -n "$OUTPUT_PATH" ]]; then
  [[ ! -e "$OUTPUT_PATH" ]] || { echo "Refusing to overwrite output: $OUTPUT_PATH" >&2; exit 3; }
fi
mkdir -p results/raw

set +e
set -o pipefail
/usr/bin/time -v "${OFFLINE_ENV[@]}" "${COMMAND[@]}" 2>&1 | tee "$LOG_PATH"
run_status=${PIPESTATUS[0]}
set -e

sha256sum "$LOG_PATH"
wc -l -c "$LOG_PATH"
printf 'RUN_EXIT_STATUS=%s\n' "$run_status"
exit "$run_status"
