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

# --- HF cache on $SCRATCH (large, persistent model weights) ---
export HF_HOME="${HF_HOME:-${SCRATCH}/huggingface}"
# reasoning-core-env's lexical_knowledge task scores via nltk (wordnet/omw); pre-download
# those corpora here once (uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')").
export NLTK_DATA="${NLTK_DATA:-${SCRATCH}/nltk_data}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH}/cache}"

# --- compile/autotune caches: node-local inside a job, else $SCRATCH ---
# Triton/inductor/vLLM autotuning writes many tiny .json cache files. On beegfs (shared
# scratch), the parallel DP inference replicas + the trainer race on them: one process
# reads an autotune entry another is still writing and hits FileNotFoundError mid-generation
# (`triton_..._0.json` in sample_tokens), which kills the vLLM worker at the first /generate.
# Node-local $SLURM_TMPDIR (NVMe) is per-node and race-free. These caches are ephemeral
# (rebuilt each job), so losing them at job end is fine — unlike HF weights above.
_COMPILE_CACHE_ROOT="${SLURM_TMPDIR:-$XDG_CACHE_HOME}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${_COMPILE_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${_COMPILE_CACHE_ROOT}/torch_inductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${_COMPILE_CACHE_ROOT}/vllm}"
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
