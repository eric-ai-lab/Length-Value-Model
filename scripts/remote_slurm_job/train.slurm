#!/bin/bash
#SBATCH --job-name=lenvm
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

START_TS=$(date +%s)


DEBUG="${DEBUG:-0}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"
ENABLE_MODULE="${ENABLE_MODULE:-0}" # TODO: change to enable module
REMOTE_BASE="${REMOTE_BASE:-/path/to/your/remote/base/${USER}}" # TODO: change to your remote base
REMOTE_REPO="${REMOTE_REPO:-$REMOTE_BASE/Length-Value-Model}" # TODO: change to your remote repo
CACHE_ROOT="${CACHE_ROOT:-$REMOTE_BASE/.cache}" # TODO: change to your cache root

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

GPU_COUNT="${SLURM_GPUS_ON_NODE:-1}"
CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((GPU_COUNT - 1))")"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_PROJECT
export WANDB_API_KEY

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME"

cd "$REMOTE_REPO"
mkdir -p logs


if [[ "$ENABLE_MODULE" -eq 1 ]]; then
  module load cuda/12.8.1
fi

# Setup environments
bash scripts/environment/env_config_uv.sh
echo "Environments setup complete"

# Download data if needed
if [[ "$DOWNLOAD_DATA" -eq 1 ]]; then
  bash scripts/data_generation/download_data.sh
fi

# Run data generation demo
DP_SIZE=$GPU_COUNT bash scripts/data_generation/demo.sh
echo "Data generation demo complete"

# Run training demo
bash scripts/training/demo.sh
echo "Training demo complete"

# Run inference demo (LIFEBench, Tradeoff Curve, Length Prediction, Length Token Wordcloud)
DP_SIZE=$GPU_COUNT bash scripts/inference/demo_lifebench.sh
DP_SIZE=$GPU_COUNT bash scripts/inference/demo_tradeoff.sh
DP_SIZE=$GPU_COUNT bash scripts/inference/demo_length_prediction.sh
DP_SIZE=$GPU_COUNT bash scripts/inference/demo_length_token.sh
DP_SIZE=$GPU_COUNT bash scripts/visualization/demo_visual.sh

echo "Inference demo complete"

END_TS=$(date +%s)
echo "Total runtime: $(((END_TS - START_TS) / 60)) minutes"
