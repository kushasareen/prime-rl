#!/usr/bin/env bash
#SBATCH --job-name=prime-rl
#SBATCH --account=aip-siamakx
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --nodes=1
#SBATCH --time=2:59:00
#SBATCH --output=logs/slurm-%j.out
#
# Integrated RL run (trainer + orchestrator + inference) on one h100:4 node.
#
# The `rl` launcher places inference and the trainer on SEPARATE physical GPUs,
# so a single-GPU node cannot run it — hence h100:4. The 4 GPUs are split as
# NUM_INFER_GPUS (inference / vLLM) + NUM_TRAIN_GPUS (FSDP trainer).
#
# TAMIA NOTES (vs Mila):
#   * Alliance requires --account; GPUs are h100 (4/node) or h200 (8/node).
#   * Compute nodes have NO internet. wandb runs offline (scripts/tamia/env.sh
#     sets WANDB_MODE=offline under SLURM); `wandb sync` from the login node
#     afterwards. Models must be pre-downloaded on the login node (HF_HUB_OFFLINE
#     is set inside the job so a missing weight errors instead of hanging).
#   * VENV STAGING is OFF by default. Mila staged the venv onto node-local NVMe to
#     dodge beegfs concurrent-import ENOENTs. On Tamia $SLURM_TMPDIR is tmpfs
#     (RAM-backed), so staging a ~15G venv costs ~15G of RAM — and Lustre tends to
#     handle concurrent reads better than beegfs. Run off the Lustre venv first;
#     only set STAGE_VENV=1 if you actually hit random ENOENT import errors.
#
# Usage (submit from the repo root so relative paths resolve):
#   sbatch scripts/tamia/rl.sh                                   # reverse-text debug, 3 infer + 1 train
#   CONFIG=configs/reverse_text.toml sbatch scripts/tamia/rl.sh  # a real experiment
#   NUM_INFER_GPUS=2 NUM_TRAIN_GPUS=2 sbatch scripts/tamia/rl.sh
#
# Extra args after the script name pass through to `uv run rl`, e.g.:
#   sbatch scripts/tamia/rl.sh --max-steps 50 --wandb.name my-run
#
# Tunables via env (defaults in parens):
#   CONFIG (configs/debug/reverse_text_v1.toml)  NUM_INFER_GPUS (3)  NUM_TRAIN_GPUS (1)
#   RUN_NAME (<config>-j<jobid>)  OUTPUT_DIR ($SCRATCH/prime-rl/outputs/<RUN_NAME>)
#   STAGE_VENV (0)  set STAGE_VENV=1 to rebuild the venv on node-local tmpfs (costs RAM)
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source scripts/tamia/env.sh

CONFIG="${CONFIG:-configs/debug/reverse_text_v1.toml}"
NUM_INFER_GPUS="${NUM_INFER_GPUS:-3}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-1}"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG%.toml}")-j${SLURM_JOB_ID:-local}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/prime-rl/outputs/${RUN_NAME}}"
mkdir -p "$OUTPUT_DIR"

# --- optional: stage venv onto node-local tmpfs (only if you hit import ENOENTs) ---
if [ "${STAGE_VENV:-0}" = "1" ] && [ -n "${SLURM_TMPDIR:-}" ]; then
  export UV_PROJECT_ENVIRONMENT="${SLURM_TMPDIR}/uv-envs"
  echo "Staging venv -> ${UV_PROJECT_ENVIRONMENT} (tmpfs, rebuild from uv cache) ..."
  t0=$SECONDS
  uv sync --offline --extra flash-attn
  uv pip install --offline --python "$UV_PROJECT_ENVIRONMENT/bin/python" -e deps/verifiers/environments/reverse_text
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
echo "venv=$UV_PROJECT_ENVIRONMENT | wandb=${WANDB_MODE:-online}"
echo "output=$OUTPUT_DIR"
echo "node=$(hostname) | gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"
echo "================================================================"

uv run --no-sync rl @ "$CONFIG" \
  --deployment.num-infer-gpus "$NUM_INFER_GPUS" \
  --deployment.num-train-gpus "$NUM_TRAIN_GPUS" \
  "${infer_args[@]}" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

echo "Done: $RUN_NAME -> $OUTPUT_DIR"
