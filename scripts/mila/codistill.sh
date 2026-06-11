#!/usr/bin/env bash
#SBATCH --job-name=codistill
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=512G
#SBATCH -c 64
#SBATCH -p short-unkillable
#SBATCH --nodes=1
#SBATCH --time=2:59:00
#SBATCH --output=logs/slurm-%j.out
#
# Staged student/teacher co-distillation (`uv run codistill`) on one a100l:4 node.
#
# The trainer holds BOTH student and teacher in memory (no per-stage disk checkpoint,
# no second inference server). Generation runs on the student vLLM server in plain rl
# mode. Default split: 3 inference GPUs + 1 trainer GPU (the trainer holds 2x 0.6B).
# See scripts/mila/env.sh / rl.sh for the venv-staging and GPU-split rationale.
#
# Usage (submit from the repo root):
#   sbatch scripts/mila/codistill.sh
#   CONFIG=configs/debug/codistill_reverse_text.toml sbatch scripts/mila/codistill.sh
#   sbatch scripts/mila/codistill.sh --trainer.rl-steps 1 --max-steps 50
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source scripts/mila/env.sh

CONFIG="${CONFIG:-configs/debug/codistill_reverse_text.toml}"
NUM_INFER_GPUS="${NUM_INFER_GPUS:-3}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-1}"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG%.toml}")-j${SLURM_JOB_ID:-local}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/prime-rl/outputs/${RUN_NAME}}"
mkdir -p "$OUTPUT_DIR"

# Stage the venv onto node-local NVMe to avoid beegfs concurrent-import crashes.
if [ "${STAGE_VENV:-1}" = "1" ] && [ -n "${SLURM_TMPDIR:-}" ]; then
  export UV_PROJECT_ENVIRONMENT="${SLURM_TMPDIR}/uv-envs"
  echo "Staging venv -> ${UV_PROJECT_ENVIRONMENT} (rebuild from uv cache) ..."
  t0=$SECONDS
  uv sync --all-extras
  echo "Staged in $((SECONDS - t0))s"
fi

infer_args=()
if [ "$NUM_INFER_GPUS" -gt 1 ]; then
  infer_args=(--inference.parallel.dp "$NUM_INFER_GPUS")
fi

echo "================================================================"
echo "codistill | run=$RUN_NAME"
echo "config=$CONFIG | infer_gpus=$NUM_INFER_GPUS train_gpus=$NUM_TRAIN_GPUS"
echo "venv=$UV_PROJECT_ENVIRONMENT"
echo "output=$OUTPUT_DIR"
echo "node=$(hostname) | gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"
echo "================================================================"

uv run codistill @ "$CONFIG" \
  --deployment.num-infer-gpus "$NUM_INFER_GPUS" \
  --deployment.num-train-gpus "$NUM_TRAIN_GPUS" \
  "${infer_args[@]}" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

echo "Done: $RUN_NAME -> $OUTPUT_DIR"
