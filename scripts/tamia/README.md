# Running prime-rl on Tamia (Digital Research Alliance)

Tamia is an Alliance/Compute-Canada cluster (Gentoo software stack, Lmod, Lustre,
SLURM). prime-rl runs here as a **uv project** — we do **not** use conda or the
Alliance wheelhouse. uv downloads its own standalone CPython and pulls wheels from
PyPI/GitHub directly, so the wheelhouse is bypassed entirely.

Hardware: GPU nodes are `h100:4` (48 CPU / 500G RAM / 4×H100-80G) and
`h200:8` (64 CPU / 1000G RAM / 8×H200). Default account: `aip-siamakx`.

## The two facts that shape everything

1. **`$HOME` has a 250k-FILE inode quota.** A full prime-rl venv blows it. So the
   venv, uv cache, uv-managed pythons, and all framework caches live on
   **`$SCRATCH`** (same Lustre mount as `$HOME`, so uv hardlinks cache→venv for
   nearly free). `scripts/tamia/env.sh` sets all of this. The repo itself stays in
   `$HOME` (it's only ~500 files).

2. **Compute nodes reach the internet only via a proxy.** No direct route, but the
   `httpproxy/1.0` module exposes a Squid proxy (`squid.tamia.ecpia.ca:3128`). Inside
   a job `env.sh` sets `http(s)_proxy` to it, so **wandb logs online** (live dashboard)
   and **HF auto-fetches** any missing files. Pre-downloading big models/datasets on
   the login node is still recommended — pulling multi-GB weights through the proxy
   mid-job wastes walltime. `env.sh` still pins `UV_NO_SYNC=1` (never sync at runtime).
   If the proxy is ever down, fall back with `WANDB_MODE=offline HF_HUB_OFFLINE=1 sbatch ...`.

> `$SCRATCH` is purged after 60 days of no access. The venv is fully regenerable
> (`uv sync`); model weights you want to keep long-term belong in `/project`.

## One-time setup (login node)

```bash
# 1. uv must be >= 0.11.1 (the project enforces it; older uv silently ignores the
#    lockfile and the exclude-newer cooldown). Upgrade if needed:
uv self update

# 2. Init submodules (skips the private configs submodule):
bash scripts/install.sh        # or: git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config

# 3. Build the venv on $SCRATCH (uses env.sh redirects). Run on the LOGIN node:
source scripts/tamia/env.sh
uv sync --extra flash-attn
# reverse_text isn't its own extra (only the all-or-nothing `envs` extra), so install
# it editably into the project venv. NOTE: a later bare `uv sync` will PRUNE it (it's
# not in the lock) — re-run this line if that happens, or use `--extra envs` for all envs.
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" -e deps/verifiers/environments/reverse_text
```

Add `source ~/path/to/prime-rl/scripts/tamia/env.sh` (or just the `export`s) to your
`~/.bashrc` so interactive `uv run` picks up the scratch venv automatically.

## Pre-download models AND datasets (login node)

Compute nodes can't reach HuggingFace. Fetch BOTH the model weights and every
dataset the env loads, into the scratch HF cache, before submitting. A missing one
fails inside the job with `Couldn't reach '<repo>' on the Hub (OfflineModeIsEnabled)`.

> **Always run `hf download` — do NOT just `ls` the cache dir.** A model dir can
> exist with only the tokenizer files cached (from prior tokenizer-only use) and no
> weights; `ls` "sees" it but vLLM then dies offline with `LocalEntryNotFoundError`.
> `hf download` is idempotent — it no-ops if complete, fetches what's missing otherwise.

```bash
source scripts/tamia/env.sh

# Model: `hf download` is enough (from_pretrained reads the hub snapshot cache).
uv run hf download PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT

# Dataset: `hf download` is NOT enough. `datasets.load_dataset()` in offline mode
# needs the library's *generated Arrow cache*, which only exists after you actually
# run load_dataset once while online. So build it:
uv run python -c "from datasets import load_dataset; load_dataset('PrimeIntellect/Reverse-Text-RL', split='train')"
```

The dataset id + split live in the env package, e.g.
`deps/verifiers/environments/reverse_text/reverse_text_v1.py` →
`dataset_name = "PrimeIntellect/Reverse-Text-RL"`, `dataset_split = "train"`. Check
the env you're training on (and pre-build every split it uses).

## Submit a job

```bash
sbatch scripts/tamia/rl.sh                                    # reverse-text RL debug
sbatch scripts/tamia/codistill.sh                             # co-distillation
CONFIG=configs/reverse_text.toml sbatch scripts/tamia/rl.sh   # a real experiment
NUM_INFER_GPUS=2 NUM_TRAIN_GPUS=2 sbatch scripts/tamia/rl.sh
```

Extra args after the script name pass through to `uv run rl`/`codistill`:

```bash
sbatch scripts/tamia/rl.sh --max-steps 50 --wandb.name my-run
```

Interactive debugging on a GPU node:

```bash
salloc --account=aip-siamakx --gres=gpu:h100:4 --cpus-per-task=48 --mem=498G --time=2:59:00
source scripts/tamia/env.sh
uv run rl @ configs/debug/reverse_text_v1.toml --deployment.num-infer-gpus 3 --deployment.num-train-gpus 1
```

## wandb

Jobs log **online** through the proxy — watch them live on the dashboard (project set in
the config). No manual sync needed. If you ran with `WANDB_MODE=offline` (proxy down),
push afterwards from the login node: `wandb sync $SCRATCH/wandb/offline-run-*`.

## Troubleshooting

- **Random `FileNotFoundError` on torch/*.py during concurrent import** — the
  network-FS concurrent-import race seen on Mila's beegfs. Re-run with
  `STAGE_VENV=1 sbatch ...` to rebuild the venv on node-local tmpfs first. Note
  `$SLURM_TMPDIR` is tmpfs (RAM) on Tamia, so this costs ~15G of RAM.
- **`uv` ignores the lockfile / TOML parse warning on `exclude-newer`** — your uv is
  older than 0.11.1. `uv self update`.
- **Job hangs reaching huggingface.co / api.wandb.ai** — you're online-mode on a
  compute node. Confirm `env.sh` was sourced (`echo $WANDB_MODE` should print
  `offline` inside a job) and that the model was pre-downloaded on the login node.
- **`LocalEntryNotFoundError` / `couldn't connect to huggingface.co` during inference
  startup** — the model is only *partially* cached (e.g. tokenizer files but no
  weights). Run `uv run hf download <model>` on the login node; don't trust an `ls`.
- **`ConnectionError: Couldn't reach '<repo>' on the Hub (OfflineModeIsEnabled)`** —
  a model or dataset wasn't pre-cached. For a dataset, note that `hf download
  --repo-type dataset` is NOT enough: run `load_dataset(...)` once online to build
  the Arrow cache (see "Pre-download models AND datasets"). The failure often
  appears late (mid-rollout) because `load_tasks` loads the dataset lazily, so
  "Train environment(s) ready" does not mean the dataset actually loaded.
- **`Failed to generate package metadata for vllm ... aarch64.whl: tcp connect error`** —
  `uv run` tried to sync on a compute node. The scripts use `uv run --no-sync` and
  `env.sh` sets `UV_NO_SYNC=1` under SLURM; make sure `env.sh` was sourced.
- **`Failed to parse environment variable UV_NO_SYNC ... expected a boolish value`** —
  an empty `UV_NO_SYNC=""` leaked onto the login node. `env.sh` only exports it
  inside a job; re-source the current `env.sh`.
- **`$HOME` quota exceeded mid-install** — something wrote to `$HOME` instead of
  `$SCRATCH`. Check `UV_PROJECT_ENVIRONMENT`/`UV_CACHE_DIR`/`HF_HOME` point at
  `$SCRATCH` (`diskusage_report` to inspect quotas).
```
