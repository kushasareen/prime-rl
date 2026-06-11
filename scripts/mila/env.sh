#!/usr/bin/env bash
# Mila-cluster environment for prime-rl, sourced by scripts/mila/*.sh.
#
# prime-rl is a uv project (NOT conda). The torch/vLLM/flash-attn wheels bundle
# their own CUDA runtime, so no `module load cuda` is needed — the interactive
# debug runs (sft/trainer/inference) all passed module-free. We only need the
# NVIDIA driver, which is present on every GPU node.
#
# Override any value by exporting it before `sbatch`, e.g.:
#   CONFIG=configs/reverse_text.toml sbatch scripts/mila/rl.sh

# --- scratch root (Mila convention) ---
export SCRATCH="${SCRATCH:-/network/scratch/${USER:0:1}/${USER}}"

# --- uv: shared env + cache on $SCRATCH (keeps writes off $HOME disk-quota) ---
# These mirror ~/.bashrc so the job is self-contained even if .bashrc isn't sourced.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH}/uv/cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${SCRATCH}/uv/envs}"

# --- HF + framework caches on $SCRATCH ---
export HF_HOME="${HF_HOME:-${SCRATCH}/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH}/cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torch_inductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${XDG_CACHE_HOME}/vllm}"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

# --- torch.compile / inductor: compile in-process (no subprocess worker pool) ---
# The trainer triggers torch.compile internally. Inductor's parallel compile-worker
# subprocess pool intermittently fails to import torch files off beegfs
# (BackendCompilerFailed -> FileNotFoundError on an existing torch/*.py). Forcing a
# single in-process compile thread sidesteps the subprocess pool entirely.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

# --- wandb (kusha's entity is the underscore form, not the displayed hyphen) ---
# Already logged in via ~/.netrc. Set WANDB_MODE=disabled before sbatch to skip.
export WANDB_ENTITY="${WANDB_ENTITY:-kusha_sareen}"

export PYTHONUNBUFFERED=1
