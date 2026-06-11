# prime-rl — Codebase Overview

A practical map of the codebase for someone who hasn't read it. Written against the
clone at `/network/scratch/k/kusha.sareen/prime-rl` (commit `e0f8a35`). File paths are
relative to the repo root; line numbers drift, so grep the named function if it moved.

---

## 1. The big picture: three processes, one loop

prime-rl trains LLMs with RL by running **three cooperating processes** that talk over
HTTP / a shared filesystem / ZMQ. This separation is the single most important thing to
understand — almost everything else is a detail of one of these three.

```
                    ┌──────────────────────────────────────────────────┐
                    │                  uv run rl                        │
                    │  (entrypoints/rl.py — supervises the 3 below)     │
                    └──────────────────────────────────────────────────┘
                                         │ launches + monitors
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                 ▼                                 ▼
┌───────────────┐  rollout requests ┌───────────────┐   training batches  ┌───────────────┐
│   INFERENCE   │◀──────────────────│ ORCHESTRATOR  │────────────────────▶│    TRAINER    │
│  (vLLM server)│   (OpenAI HTTP)   │ (rollouts +   │  (filesystem / ZMQ) │  (FSDP2, the  │
│               │──────────────────▶│  advantages)  │                     │   loss/optim) │
│   GPU(s) A    │   completions     │   (CPU, async)│◀────────────────────│   GPU(s) B    │
└───────────────┘                   └───────────────┘   updated weights   └───────────────┘
        ▲                                                  (broadcasts/ dir or NCCL)
        └──────────────────────────────────────────────────────────────────────┘
                    trainer pushes new policy weights back to inference

```

**The loop, one step at a time:**

1. **Orchestrator** pulls prompts from an *environment* and asks the **inference** server
   to generate completions (rollouts) — `group_size` samples per prompt.
2. The **environment** (a `verifiers` env) scores each rollout → a scalar **reward**.
3. **Orchestrator** turns rewards into **advantages** (GRPO: subtract the group mean),
   runs quality **filters**, packs a `batch_size` batch of `TrainingSample`s, and ships it
   to the trainer.
4. **Trainer** computes the **policy-gradient loss** (importance-weighted, with a DPPO
   trust-region mask + KL penalty), does an FSDP2 backward + optimizer step.
5. **Trainer** broadcasts the updated weights back to **inference** (via a shared
   `broadcasts/` dir or NCCL). Inference hot-reloads them via `/update_weights`.
6. Repeat. The orchestrator runs *ahead* of the trainer asynchronously (up to
   `max_off_policy_steps`), so generation and training overlap.

Key consequence you already hit: **the `rl` launcher puts inference and the trainer on
separate physical GPUs**, so a single-GPU box can't run an integrated RL job — you need
≥2 GPUs (we use `a100l:4`, split 1 inference + 3 trainer).

---

## 2. Repository layout

```
src/prime_rl/
├── entrypoints/        # CLI entry points: rl, trainer, orchestrator, inference, sft
├── trainer/            # The training process (loss, FSDP, optim, checkpoint)
│   ├── rl/             #   RL trainer: train.py (loop), loss.py (THE loss math)
│   ├── sft/            #   SFT trainer: train.py, data.py
│   ├── models/         #   model definitions / loaders
│   ├── distributed/    #   expert-parallel / deepep plumbing
│   ├── model.py        #   model load + FSDP2 setup
│   ├── optim.py        #   adamw / sgd / muon / signsgd
│   ├── ckpt.py         #   distributed checkpoint (DCP)
│   └── batch.py        #   rollout → packed micro-batch
├── orchestrator/       # The rollout-generation process (CPU, asyncio)
│   ├── orchestrator.py #   main_loop()
│   ├── dispatcher.py   #   schedules rollouts, off-policy aging
│   ├── train_sink.py   #   tokenize → advantage → filter → batch assembly
│   ├── advantage.py    #   GRPO advantages + per-env strategy + length penalty
│   ├── filters.py      #   gibberish / repetition / zero-advantage filters
│   ├── envs.py         #   wraps `verifiers` environments
│   ├── watcher.py      #   polls for new weights, calls inference /update_weights
│   └── env_server/     #   ZMQ server hosting env worker processes
├── inference/          # The vLLM server + prime-rl extensions
│   ├── patches.py      #   vLLM monkey-patches (registered as a vllm plugin)
│   └── vllm/
│       ├── server.py   #   custom routes (/update_weights, /pause, ...)
│       └── worker/     #   weight-update workers (filesystem.py, nccl.py)
├── transport/          # Orchestrator→trainer batch transport (filesystem / ZMQ)
├── utils/              # config cli(), logging, monitor (wandb), pathing
└── templates/          # Jinja2 SLURM sbatch templates

packages/prime-rl-configs/   # ALL pydantic config classes live here (not in src/)
deps/                        # git submodules: verifiers (envs), renderers, pydantic-config
configs/                     # TOML experiment configs (debug/ + real ones)
scripts/mila/                # ← our Mila SLURM scripts (env.sh, rl.sh)
docs/                        # upstream docs: training.md, scaling.md, inference.md, ...
```

A non-obvious quirk: **config classes are NOT under `src/`** — they're in
`packages/prime-rl-configs/src/prime_rl/configs/`. If you're looking for what a knob does,
that's where to grep.

---

## 3. Inference (vLLM server)

**Entry:** `src/prime_rl/entrypoints/inference.py` → `src/prime_rl/inference/vllm/server.py`.
It's a normal **vLLM OpenAI-compatible server** (`/v1/chat/completions`, `/v1/completions`,
`/v1/models`) plus prime-rl-specific routes for the RL loop:

| Route | Purpose |
|---|---|
| `POST /update_weights` | Hot-reload policy weights from the trainer |
| `POST /init_broadcaster` | Set up the NCCL receiver (NCCL mode only) |
| `POST /pause`, `/resume` | Pause/resume generation without dropping KV cache |
| `GET /liveness` | Engine health via worker RPC |

**Weight hot-reload** is the interesting part (`inference/vllm/worker/`):
- **`filesystem.py`** (default): `update_weights_from_path()` reads an HF-format checkpoint
  the trainer wrote to a shared dir and reloads it layer-by-layer
  (`load_weights_checkpoint_layerwise`). No GPU-to-GPU wiring needed → works for any
  deployment. This is what the debug runs use.
- **`nccl.py`**: receives weights directly over NCCL from the trainer's broadcaster
  (lower latency, needs the inference GPUs in the trainer's process group).

`src/prime_rl/inference/patches.py` is registered as a vLLM plugin
(`[project.entry-points."vllm.general_plugins"]` in `pyproject.toml`) so it runs in **every**
vLLM process. It's a stack of monkey-patches for transformers-v5 compat, LoRA key handling,
fp32 lm-head, prompt-length validation, MoE/routed-expert export, etc. Treat it as the
"make vLLM do RL things" shim.

**Deployment shapes** (`configs/.../inference.py`): `single_node` (default), `multi_node`
(replicas behind a router, or wide expert-parallel), and `disaggregated` (separate
prefill/decode node pools with NIXL KV transfer). `inference.parallel.tp`/`dp` set vLLM
tensor/data parallelism; `num_infer_gpus` must equal `tp * dp`.

---

## 4. Orchestrator (rollouts, rewards, advantages)

**Entry:** `src/prime_rl/orchestrator/orchestrator.py` → `main_loop()`. This process is
**CPU-only and fully async** — it's a pipeline of asyncio tasks, not a GPU job.

The pipeline (`dispatcher.py` → `train_sink.py`):

1. **Dispatcher** (`dispatcher.py`) opens "groups" (one prompt × `group_size` rollouts),
   calls the env to run rollouts against the inference server, and keeps up to
   `max_inflight` in flight. It respects a **dispatch gate** that pauses generation when the
   orchestrator gets more than ~1 step ahead of the current policy version.
2. **TrainSink** (`train_sink.py`) processes finished rollouts in three levels:
   - `process_rollout`: tokenize the prompt+completion (eagerly).
   - `process_group`: drop errored rollouts, **assign advantages** for the group, run
     **pre-batch filters**.
   - `process_batch`: once `batch_size` rollouts accrue, run **post-batch filters** and
     emit a `TrainBatch`.
3. **`finalize_train_batch`** (orchestrator.py): save rollouts to disk, (OPD only) fetch
   teacher logprobs, and `sender.send(TrainingBatch)` to the trainer.

**Advantages** (`advantage.py`) — this is the "what does the trainer learn from" signal:
- Default is **GRPO**: `advantages = rewards - rewards.mean()` within the group
  (`default_advantage_fn`). No learned value function.
- Optional **length-penalty / efficiency shaping** (`_efficiency_shaping`) rewards shorter
  correct answers.
- **Per-env advantage strategy** (the recent `e0f8a35` commit): each training env can have
  its own advantage function; `assign_advantages()` is called per-group with that env's fn.
- The advantage is a **single scalar per rollout**, later broadcast to every completion
  token in the trainer.

**Filters** (`filters.py`) — guard against degenerate training signal. Each runs in
`enforce` (drop the rollout) or monitor-only mode:
- `GibberishFilter`, `RepetitionFilter` — detect broken generations.
- `ZeroAdvantageFilter` (enforced post-batch) — drops rollouts whose advantage is 0 (i.e.
  every sample in the group got the same reward → no gradient signal). **This is the source
  of the "10 consecutive zero-trainable batches" error** you saw when running the
  standalone orchestrator against the base model: a too-weak model scores 0 on everything,
  so every group is uniform and gets filtered out.

**Environments** (`envs.py`) wrap the `verifiers` library (in `deps/verifiers`). An env
supplies prompts and a **rubric** that scores rollouts into rewards. Envs run in separate
worker processes behind a ZMQ server (`env_server/`). Multi-env training is weighted
round-robin over `[[orchestrator.train.env]]` entries; eval envs
(`[[orchestrator.eval.env]]`) fire every `interval` steps and report pass@k.

**Off-policy coordination** (`watcher.py`): a `WeightWatcher` task polls the `broadcasts/`
dir for a new trainer checkpoint, calls inference `/update_weights`, bumps `policy.version`,
and ages in-flight rollouts. Rollouts older than `max_off_policy_steps` (default 8) are
cancelled. This is what lets generation run ahead of training without going stale.

---

## 5. Trainer + the loss (the core you asked about)

**Entry:** `src/prime_rl/trainer/rl/train.py` → `train()`. Launched under `torchrun`
(`--nproc-per-node = num_train_gpus`). The loop per step:

1. Broadcast current weights to inference (`broadcast/{filesystem,nccl}.py`).
2. Wait for a `TrainingBatch` from the orchestrator (via `transport/`).
3. Pack into micro-batches (`trainer/batch.py`, `trainer/rl/packer.py`) — FFD bin-packing
   of variable-length sequences to minimize padding.
4. Forward + **`compute_loss`** + backward over micro-batches.
5. Gradient-norm rescale to a **global** token count, optional clip, optimizer step,
   scheduler step.

### The loss — `src/prime_rl/trainer/rl/loss.py`

The mode dispatches to one of three functions; RL uses **`default_loss_fn`** — an
importance-sampled policy gradient with a DPPO trust-region mask and a KL penalty (a
"DPPO+KL" formulation, not vanilla PPO-clip):

```python
# importance ratio between the trainer's current policy and the inference policy
# that actually generated the tokens (the rollouts are slightly off-policy):
log_importance_ratio = trainer_logprobs - inference_logprobs
importance_ratio     = torch.exp(log_importance_ratio)

# DPPO trust region: instead of PPO's ratio clip, mask out tokens whose *probability*
# moved too far in the direction the advantage pushes (asymmetric high/low thresholds):
probs_diff = exp(trainer_logprobs) - exp(inference_logprobs)
dppo_invalid = where(advantages > 0, probs_diff >  dppo_mask_high,
                                     probs_diff < -dppo_mask_low)
keep_mask = loss_mask & ~dppo_invalid

# policy-gradient term + KL regularizer toward the inference/reference policy:
pg_loss = keep_mask * (adv_tau * advantages) * importance_ratio
kl_loss = loss_mask * log_importance_ratio**2
loss    = (-pg_loss + kl_tau * kl_loss).sum()
```

The knobs (in `configs/trainer.py`, `DefaultLossConfig`):
- `dppo_mask_low` / `dppo_mask_high` (default 0.2) — the asymmetric trust-region thresholds.
- `adv_tau` (default 1.0) — advantage temperature/scale.
- `kl_tau` (default 1e-3) — KL penalty weight.

Important mechanics:
- **Advantages are per-rollout scalars** broadcast to every completion token
  (`batch.py`: `advantages = [sample.advantage] * len(input_ids)`); the loss is otherwise
  **per-token**.
- `inference_logprobs` come from the orchestrator (the policy that generated the tokens),
  `trainer_logprobs` are recomputed in the forward pass — their ratio is the off-policy
  correction.
- Loss is summed and divided by the **global** (all-rank) token count so gradient scale is
  invariant to how sequences happened to pack across ranks.

The other two modes share this file:
- **`opd_loss_fn`** (on-policy distillation): KL to a *teacher's* logprobs instead of a
  reward-derived advantage.
- **`sft_loss_fn`**: plain masked negative-log-likelihood (`-trainer_logprobs[loss_mask].sum()`).

### Shared trainer machinery
- **`model.py`** — loads the HF model to meta device, then `fully_shard` (FSDP2) over the
  data-parallel mesh; supports HSDP (`dp_replicate`), expert parallel (`ep`), context
  parallel (`cp`), CPU offload, mixed precision.
- **`optim.py`** — adamw (default), sgd, **muon**, sign_sgd; optional optimizer-state CPU
  offload.
- **`ckpt.py`** — `torch.distributed.checkpoint` (DCP) sharded checkpoints that can resume
  at a different world size. Writes a `STABLE` marker when complete.

### SFT trainer
`src/prime_rl/trainer/sft/` is the same FSDP/checkpoint machinery with a cross-entropy loss
and prompt/completion **loss masking** (`sft/data.py`) and sequence packing. It's standalone
(`uv run sft`, wraps torchrun itself) and is the usual warm-start before RL.

---

## 6. Weight sync: trainer → inference

Two transports, chosen by `weight_broadcast.type`:
- **`filesystem`** (default, what debug uses): trainer gathers weights on rank 0, writes an
  HF checkpoint to a shared dir + a `STABLE` marker; the orchestrator's watcher tells
  inference to `update_weights_from_path`. Slow-ish but works anywhere.
- **`nccl`**: trainer broadcasts tensors directly to the inference GPUs over NCCL. Fast, but
  requires the inference GPUs to join the trainer's process group (`/init_broadcaster`).

`TrainingSample` (the orchestrator→trainer payload, `transport/types.py`) carries
`prompt_ids`, `completion_ids`, `completion_logprobs` (= the `inference_logprobs` above),
per-token temperatures, the scalar `advantage` and `reward`, `env_name`, and (OPD only)
`teacher_logprobs`.

---

## 7. The config system

prime-rl uses **pydantic-config** (`deps/pydantic-config`). Every process is configured by a
single pydantic model resolved from TOML files + CLI overrides via `cli(SomeConfig)`.

- **`@ file.toml`** loads a TOML into the config. **`--a.b.c value`** overrides a nested
  field. They compose and deep-merge:
  ```bash
  uv run rl @ configs/debug/reverse_text_v1.toml \
      --max-steps 50 --trainer.optim.lr 1e-5 \
      --deployment.num-infer-gpus 1 --deployment.num-train-gpus 3
  ```
- `RLConfig` is the merged root for `uv run rl`. It contains `trainer`, `orchestrator`, and
  (optional) `inference` sub-configs, plus **shared** fields that *propagate down*: setting
  top-level `model.name`, `max_steps`, `seq_len`, `wandb.*`, `ckpt.*` fills the matching
  field in all three sub-configs (so you don't repeat the model name three times). Setting a
  field at both levels is fine **only if the values match**.
- A real config looks like (`configs/hendrycks_math/rl.toml`):
  ```toml
  max_steps = 500
  seq_len   = 2048
  [model]            name = "Qwen/Qwen3-4B-Instruct-2507"
  [orchestrator]     batch_size = 512  group_size = 16
  [orchestrator.train.sampling]   max_completion_tokens = 2048
  [[orchestrator.train.env]]      id = "math-env"  name = "hendrycks-math"
                                  args = { dataset_name = "PrimeIntellect/Hendrycks-Math", ... }
  [orchestrator.eval]             interval = 10
  [[orchestrator.eval.env]]       id = "math500"  num_examples = 30  group_size = 4
  [trainer]   # inherits defaults
  [inference] # inherits defaults
  ```
- Empty tables like `[ckpt]` / `[inference]` mean "enable this with default settings".

Config classes: `packages/prime-rl-configs/src/prime_rl/configs/{rl,trainer,orchestrator,inference,shared}.py`.

---

## 8. How experiments are run

**Entry points** (`pyproject.toml [project.scripts]`):

| Command | What it does |
|---|---|
| `uv run rl @ cfg.toml` | The main one. Launches + supervises inference + orchestrator + trainer. |
| `uv run sft @ cfg.toml` | Standalone SFT (wraps torchrun itself). |
| `uv run trainer @ cfg.toml` | The RL trainer **alone** — must be launched under `torchrun` (it expects `RANK`); use only when wiring processes manually. |
| `uv run orchestrator @ cfg.toml` | The orchestrator alone (point it at a running inference server). |
| `uv run inference @ cfg.toml` | The vLLM server alone. |

**`uv run rl` (`entrypoints/rl.py`)** is the conductor:
1. `cli(RLConfig)` resolves the config, then `write_subconfigs()` dumps materialized
   `trainer.toml` / `orchestrator.toml` / `inference.toml` into `<output_dir>/configs/`.
2. **GPU split**: `num_infer_gpus` GPUs (ids `0..n-1`) go to inference, the next
   `num_train_gpus` go to the trainer. Total must be ≤ physical GPUs — *this is why a
   1-GPU node fails with "Requested 2 GPUs … only 1 available"*.
3. Launches three subprocesses: `inference @ ...`, `orchestrator @ ...`, and
   `torchrun --nproc-per-node=<num_train_gpus> -m prime_rl.trainer.rl.train @ ...`.
4. Monitors them; if the trainer or orchestrator dies, it tears everything down.
5. If `[slurm]` is set, instead renders a Jinja sbatch from `src/prime_rl/templates/*.j2`
   and `sbatch`es it (the multi-node path).

**`--dry-run`** validates the config and writes the subconfigs *without* launching or
checking GPUs — handy for sanity-checking a config from a login node.

### Running on Mila (our scripts)
We don't use prime-rl's built-in `[slurm]` integration; we submit a normal sbatch that runs
`uv run rl` locally on one multi-GPU node. See `scripts/mila/`:
- **`env.sh`** — sourced; sets the uv shared-env + caches (HF, triton, inductor, vLLM) on
  `$SCRATCH`, `WANDB_ENTITY`, and the inductor fix (see Gotchas). No `module load cuda` —
  the wheels bundle CUDA.
- **`rl.sh`** — sbatch for `a100l:4` on `short-unkillable` (max 3 h). Splits the 4 GPUs as
  `NUM_INFER_GPUS` (default 1) + `NUM_TRAIN_GPUS` (default 3). Parameterized:
  ```bash
  sbatch scripts/mila/rl.sh                                     # reverse-text debug
  CONFIG=configs/hendrycks_math/rl.toml sbatch scripts/mila/rl.sh
  NUM_INFER_GPUS=2 NUM_TRAIN_GPUS=2 sbatch scripts/mila/rl.sh
  sbatch scripts/mila/rl.sh --max-steps 50 --wandb.name my-run  # extra args pass through
  ```

### Training modes
`orchestrator.training_mode`:
- **`rl`** — standard: student generates, env rewards, policy-gradient loss.
- **`opd`** (on-policy distillation): student generates, trainer minimizes KL to a *teacher*
  inference server's logprobs. Needs `[orchestrator.teacher]` (must be vLLM, for
  `prompt_logprobs`).
- **`sft`** — teacher generates the rollouts, student is trained on them as supervised data.

---

## 9. Outputs, logging, resume

A run writes under `--output-dir` (we set `$SCRATCH/prime-rl/outputs/<run>`):
```
<output_dir>/
├── configs/         # materialized trainer.toml / orchestrator.toml / inference.toml
├── logs/            # trainer.log, orchestrator.log, inference.log, trainer/torchrun/*, envs/*
├── checkpoints/     # trainer DCP checkpoints, step_N/ with a STABLE marker
├── rollouts/        # orchestrator rollout batches (the off-policy buffer)
├── broadcasts/      # weight snapshots handed to inference
└── run_default/     # orchestrator's own state dir (+ its wandb)
```

- **wandb**: `uv run rl` runs trainer + orchestrator in **shared mode** logging to *one* run
  (set `WANDB_ENTITY`; already logged in via `~/.netrc`). The metric lines you watch in the
  log — `Step N | Reward … | Trainable …/… | Truncation …` — come from the orchestrator.
- **Resume**: set `ckpt.resume_step = N` (or `-1` for latest). Absent `[ckpt]` = from
  scratch. `--clean-output-dir` wipes first.
- **Where to look when something breaks**: env errors → `logs/envs/.../*.log`; trainer
  crashes / OOM → `logs/trainer.log` (and per-rank `logs/trainer/torchrun/.../stderr.log`);
  generation issues → `logs/inference.log`.

---

## 10. Gotchas (things that bit us)

- **≥2 GPUs required for `uv run rl`.** Inference and trainer get *disjoint* physical GPUs;
  a 1-GPU node errors out. The individual pieces (`sft`, `inference`, standalone `trainer`
  under torchrun) each run on 1 GPU fine.
- **`uv run trainer` needs torchrun.** Bare `uv run trainer @ ...` dies with
  `environment variable RANK expected` — it's the standalone trainer, meant to be launched
  by `torchrun` (or by `uv run rl`, which does that for you).
- **`torch.compile` / inductor vs. beegfs.** The RL trainer triggers `torch.compile`
  internally (even with `model.compile=None`). Inductor's *parallel compile-worker
  subprocess pool* intermittently fails to import torch files off the beegfs network
  filesystem:
  `BackendCompilerFailed: FileNotFoundError: torch/distributed/elastic/agent/server/api.py`
  (the file exists — it's a transient network-FS lookup miss). When the trainer dies the
  orchestrator hangs forever waiting for weights. **Fix:** `TORCHINDUCTOR_COMPILE_THREADS=1`
  (compile in-process, no subprocess pool) — now set in `scripts/mila/env.sh`.
- **"zero-trainable batches".** If every rollout in a group gets the same reward (e.g. a
  weak base model that scores 0 on everything), GRPO advantages are all 0 and the
  `zero_advantage` filter drops them; after 10 such batches the orchestrator aborts. Use a
  model that can sometimes succeed (the reverse-text debug uses an SFT'd checkpoint for
  exactly this reason).
- **uv env is shared.** `UV_PROJECT_ENVIRONMENT=$SCRATCH/uv/envs` is a single env across uv
  projects (set in `~/.bashrc`); prime-rl's deps were synced into it.

---

## 11. Where to start reading, by question

| You want to understand… | Read |
|---|---|
| The whole loop | `orchestrator/orchestrator.py:main_loop`, `trainer/rl/train.py:train` |
| The loss math | `trainer/rl/loss.py:default_loss_fn` + `configs/trainer.py:DefaultLossConfig` |
| How rewards become advantages | `orchestrator/advantage.py`, `orchestrator/train_sink.py:process_group` |
| How rollouts are generated | `orchestrator/dispatcher.py`, `orchestrator/envs.py` |
| How weights get to inference | `trainer/rl/broadcast/`, `inference/vllm/worker/`, `orchestrator/watcher.py` |
| How a run is launched | `entrypoints/rl.py:rl_local` |
| What a knob does | `packages/prime-rl-configs/src/prime_rl/configs/*.py` |
| Upstream docs | `docs/training.md`, `docs/scaling.md`, `docs/inference.md`, `docs/algorithms.md` |
