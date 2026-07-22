#!/usr/bin/env bash
# Chain N dependent SLURM jobs that resume from ONE shared checkpoint dir, so a long run
# survives short-unkillable's 3h wall-clock limit. Job 1 starts fresh; each later job
# waits (afterany, so a wall-clock timeout still advances the chain) for the previous job
# to end, then resumes from the latest checkpoint. `--ckpt.resume-step -1` is a no-op on
# the first job (no checkpoint exists yet -> starts fresh).
#
# Requires the launcher to support checkpoint/resume: scripts/mila/rl.sh (always) and
# scripts/mila/codistill.sh (via the codistill trainer's ckpt/resume). RUN_NAME MUST be
# fixed across the chain — it determines OUTPUT_DIR, hence the shared checkpoint dir.
#
# Env:
#   LAUNCHER         scripts/mila/rl.sh | scripts/mila/codistill.sh   (required)
#   CONFIG           path to the .toml config                          (required)
#   RUN_NAME         fixed run name shared by every job in the chain   (required)
#   NUM_JOBS         how many jobs to chain (default 2)
#   CKPT_INTERVAL    save a checkpoint every N stages/steps (default 25)
#   AFTER            optional jobid; make job 1 also wait for it (to extend an existing chain)
#   NUM_INFER_GPUS / NUM_TRAIN_GPUS / OUTPUT_DIR   forwarded to the launcher if set
#
# Usage:
#   LAUNCHER=scripts/mila/rl.sh CONFIG=configs/debug/rl_reasoning_core_qwen3_4b.toml \
#     RUN_NAME=rl-rc-4b NUM_JOBS=2 CKPT_INTERVAL=25 NUM_INFER_GPUS=2 NUM_TRAIN_GPUS=2 \
#     scripts/mila/chain.sh [extra passthrough args...]
#
#   # extend the same chain later (resumes from where it left off):
#   AFTER=<last_jobid> LAUNCHER=... CONFIG=... RUN_NAME=rl-rc-4b NUM_JOBS=2 scripts/mila/chain.sh
set -euo pipefail

LAUNCHER="${LAUNCHER:?set LAUNCHER=scripts/mila/rl.sh or scripts/mila/codistill.sh}"
: "${CONFIG:?set CONFIG=path/to/config.toml}"
: "${RUN_NAME:?set RUN_NAME (fixed across the chain -> shared checkpoint dir)}"
NUM_JOBS="${NUM_JOBS:-2}"
CKPT_INTERVAL="${CKPT_INTERVAL:-25}"

# Forward launcher-facing env via sbatch's default --export=ALL.
export CONFIG RUN_NAME
# Stable wandb run id per RUN_NAME so every window in the chain RESUMES one wandb run.
# Without this the rl entrypoint mints a fresh uuid each job -> a new run per window
# (fragmented metrics). Deterministic from RUN_NAME; override by pre-exporting.
export WANDB_SHARED_RUN_ID="${WANDB_SHARED_RUN_ID:-$(printf '%s' "$RUN_NAME" | md5sum | cut -c1-32)}"
[ -n "${NUM_INFER_GPUS:-}" ] && export NUM_INFER_GPUS
[ -n "${NUM_TRAIN_GPUS:-}" ] && export NUM_TRAIN_GPUS
[ -n "${OUTPUT_DIR:-}" ] && export OUTPUT_DIR

echo "Chaining $NUM_JOBS job(s) | launcher=$LAUNCHER | config=$CONFIG | run=$RUN_NAME | ckpt.interval=$CKPT_INTERVAL${AFTER:+ | after=$AFTER}"
prev="${AFTER:-}"
for ((k = 1; k <= NUM_JOBS; k++)); do
  dep=()
  [ -n "$prev" ] && dep=(--dependency=afterany:"$prev")
  jid=$(sbatch --parsable "${dep[@]}" "$LAUNCHER" \
    --ckpt.interval "$CKPT_INTERVAL" --ckpt.resume-step -1 "$@")
  echo "  job $k/$NUM_JOBS -> $jid (resumes after ${prev:-<fresh start>})"
  prev="$jid"
done
echo "Last job in chain: $prev"
echo "Watch: squeue -u \$USER   |   logs/slurm-<jobid>.out"
