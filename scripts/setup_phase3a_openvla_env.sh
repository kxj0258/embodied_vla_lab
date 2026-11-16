#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/kxj/embodied_ws/embodied_vla_lab
CONDA_SH=/home/kxj/miniforge3/etc/profile.d/conda.sh
ENV_PREFIX=/home/kxj/miniforge3/envs/evla-openvla

if [[ -e "$ENV_PREFIX" ]]; then
  echo "Refusing to overwrite existing environment: $ENV_PREFIX" >&2
  exit 2
fi

source "$CONDA_SH"
conda create -y -n evla-openvla python=3.10 pip
conda activate evla-openvla
python -m pip install --upgrade "pip==25.2"
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0" "torchvision==0.26.0"
python -m pip install -r "$ROOT/configs/env/evla_openvla_minimal_inference.txt"

HF_HUB_OFFLINE=1 WANDB_MODE=disabled python "$ROOT/scripts/check_phase3a_openvla_runtime.py"
python -m pip freeze
