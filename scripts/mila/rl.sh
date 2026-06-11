#!/usr/bin/env bash
#SBATCH --job-name=prime-rl
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=512G
#SBATCH -c 64
#SBATCH -p short-unkillable
#SBATCH --nodes=1
#SBATCH --time=2:59:00
#SBATCH --output=logs/slurm-%j.out
#
# Integrated RL run (trainer + orchestrator + inference) on one a100l:4 node.
#
# The `rl` launcher places inference and the trainer on SEPARATE physical GPUs,
# so a single-GPU node cannot run it — hence a100l:4. The 4 GPUs are split as
# NUM_INFER_GPUS (inference / vLLM) + NUM_TRAIN_GPUS (FSDP trainer).
#
# VENV STAGING: the canonical venv lives on scratch (beegfs). Multiple ranks
# importing the 15G venv off beegfs at once intermittently get FileNotFoundError
# on files that exist (a network-FS concurrency artifact — reproduced: 1 import
# fine, 16 concurrent -> random ENOENT on existing files). So at job start we
# rebuild the venv on node-local NVMe ($SLURM_TMPDIR) and run from there.
#
# Usage (submit from the repo root so relative paths resolve):
#   sbatch scripts/mila/rl.sh                                  # reverse-text debug, 3 infer + 1 train
#   CONFIG=configs/reverse_text.toml sbatch scripts/mila/rl.sh # a real experiment
#   NUM_INFER_GPUS=2 NUM_TRAIN_GPUS=2 sbatch scripts/mila/rl.sh
#
# Extra args after the script name pass through to `uv run rl`, e.g.:
#   sbatch scripts/mila/rl.sh --max-steps 50 --wandb.name my-run
#
# Tunables via env (defaults in parens):
#   CONFIG (configs/debug/reverse_text_v1.toml)  NUM_INFER_GPUS (3)  NUM_TRAIN_GPUS (1)
#   RUN_NAME (<config>-j<jobid>)  OUTPUT_DIR ($SCRATCH/prime-rl/outputs/<RUN_NAME>)
#   STAGE_VENV (1)  set STAGE_VENV=0 to skip node-local staging (use the scratch venv)
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source scripts/mila/env.sh

CONFIG="${CONFIG:-configs/debug/reverse_text_v1.toml}"
NUM_INFER_GPUS="${NUM_INFER_GPUS:-3}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-1}"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG%.toml}")-j${SLURM_JOB_ID:-local}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/prime-rl/outputs/${RUN_NAME}}"
mkdir -p "$OUTPUT_DIR"

# --- stage venv onto node-local NVMe (avoids beegfs concurrent-import ENOENTs) ---
if [ "${STAGE_VENV:-1}" = "1" ] && [ -n "${SLURM_TMPDIR:-}" ]; then
  export UV_PROJECT_ENVIRONMENT="${SLURM_TMPDIR}/uv-envs"
  echo "Staging venv -> ${UV_PROJECT_ENVIRONMENT} (rebuild from uv cache) ..."
  t0=$SECONDS
  uv sync --all-extras
  echo "Staged in $((SECONDS - t0))s"
fi

# --- multi-GPU inference: run NUM_INFER_GPUS independent DP replicas (tp stays 1).
# The config validator requires num_infer_gpus == inference.parallel.tp * dp. ---
infer_args=()
if [ "$NUM_INFER_GPUS" -gt 1 ]; then
  infer_args=(--inference.parallel.dp "$NUM_INFER_GPUS")
fi

echo "================================================================"
echo "prime-rl | run=$RUN_NAME"
echo "config=$CONFIG | infer_gpus=$NUM_INFER_GPUS train_gpus=$NUM_TRAIN_GPUS"
echo "venv=$UV_PROJECT_ENVIRONMENT"
echo "output=$OUTPUT_DIR"
echo "node=$(hostname) | gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"
echo "================================================================"

uv run rl @ "$CONFIG" \
  --deployment.num-infer-gpus "$NUM_INFER_GPUS" \
  --deployment.num-train-gpus "$NUM_TRAIN_GPUS" \
  "${infer_args[@]}" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

echo "Done: $RUN_NAME -> $OUTPUT_DIR"
