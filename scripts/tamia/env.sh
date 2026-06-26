#!/usr/bin/env bash
# Tamia-cluster (Digital Research Alliance) environment for prime-rl.
# Sourced by scripts/tamia/*.sh and meant to be sourced interactively too:
#   source scripts/tamia/env.sh
#
# prime-rl is a uv project (NOT conda, NOT the Alliance wheelhouse). The
# torch/vLLM/flash-attn wheels bundle their own CUDA runtime, so NO `module load`
# is needed — we only rely on the NVIDIA driver present on every GPU node. uv
# downloads its own standalone CPython, so the Gentoo python module is bypassed
# entirely (and with it, the wheelhouse).
#
# Two hard Alliance facts this file encodes:
#   1. $HOME has a 250k-FILE inode quota. A full venv blows it. So the venv,
#      uv cache, uv-managed pythons, and every framework cache live on $SCRATCH
#      (same Lustre mount as $HOME, so uv hardlinks cache->venv for ~free).
#   2. COMPUTE NODES HAVE NO INTERNET. All installing/downloading happens on the
#      LOGIN node. Inside a job, wandb defaults to offline (synced later) and HF
#      must read pre-downloaded weights. See scripts/tamia/README.md.
#
# Override any value by exporting it before sourcing / before `sbatch`.

# --- scratch root (set by the cluster; fallback just in case) ---
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"

# --- uv: env + cache + managed pythons on $SCRATCH (keep writes off $HOME inodes) ---
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH}/uv/cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${SCRATCH}/uv/envs/prime-rl}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${SCRATCH}/uv/python}"
# Use uv's downloaded CPython, never the Gentoo-stack python (avoids the wheelhouse).
export UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-only-managed}"

# --- HF + framework caches on $SCRATCH ---
export HF_HOME="${HF_HOME:-${SCRATCH}/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH}/cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torch_inductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${XDG_CACHE_HOME}/vllm}"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

# --- torch.compile / inductor: compile in-process (no subprocess worker pool) ---
# The trainer triggers torch.compile internally. Inductor's parallel compile-worker
# subprocesses intermittently fail to import torch files off the shared FS. Forcing
# a single in-process compile thread sidesteps the subprocess pool entirely.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

export WANDB_DIR="${WANDB_DIR:-${SCRATCH}/wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-kusha_sareen}"
mkdir -p "$WANDB_DIR"

# --- inside a SLURM job: reach the internet via Tamia's Squid proxy ---
# Compute nodes have no direct internet, but the `httpproxy/1.0` module exposes a Squid
# proxy. With it, wandb logs ONLINE (live dashboard, no offline-sync dance) and HF
# auto-fetches any missing files (self-healing if a model/dataset wasn't fully cached).
# We still pin UV_NO_SYNC — the venv is pre-built on the login node and `uv run`'s default
# sync would needlessly hit the network (incl. the aarch64 vllm metadata that errors).
# Pre-downloading big models/datasets on the login node is still faster than pulling them
# through the proxy mid-job — see README. Only set inside a job (an empty UV_NO_SYNC on the
# login node makes uv error "expected a boolish value"). Override by pre-exporting before
# sbatch, e.g. WANDB_MODE=offline / HF_HUB_OFFLINE=1 if the proxy is down.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  export http_proxy="${http_proxy:-http://squid.tamia.ecpia.ca:3128}"
  export https_proxy="${https_proxy:-http://squid.tamia.ecpia.ca:3128}"
  export no_proxy="${no_proxy:-tamia.ecpia.ca}"
  export UV_NO_SYNC="${UV_NO_SYNC:-1}"
fi

export PYTHONUNBUFFERED=1
