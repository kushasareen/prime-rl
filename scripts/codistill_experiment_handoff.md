# Co-distillation experiment — handoff

Everything needed to run (and resume) the **RL baseline vs. staged co-distillation** experiment
on `reasoning-core-env`, on either Mila or Tamia. Cluster-agnostic except where noted.

## The experiment

Compare, on the same model / env / dataset, changing only the training algorithm:

- **RL baseline** — plain GRPO. `configs/debug/rl_reasoning_core_qwen3_4b.toml` (`uv run rl`).
- **1-4-1 co-distillation** — per stage: 1 GRPO step on the student → clone into teacher →
  4 RFT (SFT) steps on the teacher over the student's *correct* generations → 1 OPD
  (on-policy-distillation) step on the student from the teacher.
  `configs/debug/codistill_reasoning_core_qwen3_4b.toml` (`uv run codistill`).

Both: **Qwen3-4B**, env `reasoning-core/reasoning-core-env`, dataset
`kushasareen/reasoning-core-l2-3-n512` (495 train / 121 eval, difficulty levels 2–3),
`seq_len=10240`, `max_completion_tokens=8192`, `batch_size=32`, `group_size=8`,
`temperature=0.6`, student `lr=3e-6`, DefaultRenderer + `think` parser. **Compared at matched
stages** (1 stage = 1 rollout batch). The 4B init sits ~44% correct on levels 2–3 (healthy
low-but-nonzero band). A 1.5B variant exists (`codistill_reasoning_core.toml`) for fast iteration.

## What was built this session

1. **Co-distillation checkpoint/resume** (`src/prime_rl/trainer/codistill/train.py`). The
   codistill trainer previously had none. It now mirrors the RL trainer, reusing
   `setup_ckpt_managers` / `ckpt.py`: saves `{student model, student optimizer, student
   scheduler, progress}` every `ckpt.interval` stages and resumes from the latest checkpoint
   when `ckpt.resume_step == -1`. Only the student is saved — the teacher is re-cloned from it
   each stage. **Validated end-to-end**: on resume both trainer *and* orchestrator load the
   latest step, the orchestrator cleans stale future rollouts/broadcasts, and training
   continues coherently.

2. **Teacher-optimizer-free-before-OPD** (`codistill/train.py`). After RFT, `del
   teacher_optimizer` + `gc.collect()` + `empty_cache()` frees the teacher's AdamW state
   before the OPD phase (it's rebuilt from the re-cloned teacher next stage anyway). This is
   what lets the 4B run **2 infer + 2 train** at `seq_len=10240`: measured train-GPU peak
   ~64 GB / 82 GB (~80%), vs ~98.5% before. (Fallback if memory is ever tight: teacher
   `[trainer.teacher_optim] type = "sign_sgd"` — zero optimizer state, and at `lr=1e-5` its
   update scale ≈ AdamW's, so no LR retune.)

3. **Chained-job wrapper** (`scripts/mila/chain.sh`). Submits N dependent SLURM jobs sharing
   one checkpoint dir so a long run survives a cluster's wall-clock limit. `afterany` (a
   timeout still advances the chain); `--ckpt.resume-step -1` on every job (no-op on the first).
   Cluster-agnostic — pass any `LAUNCHER` (works with `scripts/tamia/*.sh` too).

4. **RL baseline config** (`configs/debug/rl_reasoning_core_qwen3_4b.toml`) — matched to the 4B
   codistill config, minus the teacher/RFT/OPD machinery.

5. **Config fixes** (both codistill configs): env id `reasoning-core-env` →
   `reasoning-core/reasoning-core-env` (the bare id never auto-installs — it must contain `/`);
   dataset → HF Hub id; `max_completion_tokens` 4096 → 8192 (truncation was 40–70% after step 0,
   now ~20%); `seq_len` → 10240 (8192 completion + 2048 prompt budget; prompts over that with a
   long completion are dropped by the packer — only ~1–3% of this dataset, p99 prompt ≈ 2.5k).

## Reward scoring fix (important — do not skip on a fresh setup)

`reasoning-core-env`'s `score_answer` silently mis-scored ~19% of rollouts (correct answers →
reward 0) because scoring dependencies were missing and the verifiers rubric swallows the
exception. Auditing gold answers per task found: `inflect` missing (set/count tasks),
`nltk` wordnet/omw data missing (lexical_knowledge), and a reasoning_core task-resolver bug on
`diff_patching`/`term_unification` (AssertionError — not fixable by installing). Fixes applied:

- **`inflect`** added to `pyproject.toml` deps (the env fails to declare it). In the lock, so
  the staged job venv has it before the runtime `prime env install`.
- **nltk corpora** (`wordnet`, `omw-1.4`) pre-downloaded to `$SCRATCH/nltk_data`; `env.sh`
  (both clusters) exports `NLTK_DATA`. On a fresh machine run once:
  `uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"`.
- **`diff_patching` + `term_unification`** dropped from the dataset (re-pushed to the same HF
  repo). New size 495 train / 121 eval.

Verified: every remaining task now scores its gold answer 1.0 on both splits (0 broken / 0
weird). Re-run the per-task audit (`score_answer(gold, info)` over one example per task) if the
env version changes.

## How to run

### Mila (`short-unkillable`, a100l:4, 3h limit)

```bash
# RL baseline — 2 chained jobs (job 2 auto-resumes after job 1 hits the 3h wall)
LAUNCHER=scripts/mila/rl.sh CONFIG=configs/debug/rl_reasoning_core_qwen3_4b.toml \
  RUN_NAME=rl-rc-4b NUM_JOBS=2 CKPT_INTERVAL=25 NUM_INFER_GPUS=2 NUM_TRAIN_GPUS=2 \
  bash scripts/mila/chain.sh

# Co-distillation — same, different launcher + config
LAUNCHER=scripts/mila/codistill.sh CONFIG=configs/debug/codistill_reasoning_core_qwen3_4b.toml \
  RUN_NAME=codistill-rc-4b NUM_JOBS=2 CKPT_INTERVAL=25 NUM_INFER_GPUS=2 NUM_TRAIN_GPUS=2 \
  bash scripts/mila/chain.sh

# Extend either chain later (resumes where it left off):
AFTER=<last_jobid> LAUNCHER=... CONFIG=... RUN_NAME=<same-run-name> NUM_JOBS=2 bash scripts/mila/chain.sh
```

**Constraint:** short-unkillable's `QOSMaxMemoryPerUser` (jobs request 512 GB) means **only one
of your jobs runs at a time**. RL and codistill chains serialize — fine for an
alternating "2 RL, 2 codistill" cadence. (Worth revisiting whether 512 GB is really needed;
lowering `--mem` in `scripts/mila/*.sh` could allow two concurrent jobs.)

### Tamia (h100:4 / h200:8) — run in parallel with Mila

The same configs and `chain.sh` work; use the Tamia launchers (which carry `--account`, the
proxy env, and `env.sh`). See `scripts/tamia/README.md` and `scripts/tamia/reasoning_core_status.md`.

```bash
LAUNCHER=scripts/tamia/codistill.sh CONFIG=configs/debug/codistill_reasoning_core_qwen3_4b.toml \
  RUN_NAME=codistill-rc-4b NUM_JOBS=2 CKPT_INTERVAL=25 NUM_INFER_GPUS=2 NUM_TRAIN_GPUS=2 \
  bash scripts/mila/chain.sh
```

Tamia notes: model + dataset auto-fetch through the Squid proxy on compute nodes, but
pre-downloading on the login node is faster (`hf download Qwen/Qwen3-4B`; build the dataset
Arrow cache once online). `env.sh` already bypasses the proxy for localhost (the orchestrator↔
vLLM fix). Keep `RUN_NAME` identical across a chain (and across clusters only if they share
`$SCRATCH`, which they do not — each cluster resumes its own checkpoint dir).

## Tuning `CKPT_INTERVAL`

Pick it so a checkpoint lands ~2–3× per wall-clock window without the write dominating. 4B full
checkpoint ≈ 25 GB (RL) / larger for codistill; the DCP write took ~66s for the 1.5B (~21 GB)
on Lustre. Start at 25 stages and adjust once you see real seconds-per-stage for the 4B at
`seq_len=10240` (generation-dominated). Progress since the last checkpoint is redone on resume.

## Current status (2026-07-01)

- Checkpoint/resume + chaining: **built and validated** (1.5B resume test resumed trainer +
  orchestrator from the latest step and finished).
- RL baseline chain submitted on Mila (`rl-rc-4b`, 2 jobs): job 1 booted cleanly (env installed,
  orchestrator loop started, `max_steps=10000`) — first rollout generating at time of writing.
- Codistill 4B at 2-2 / seq 10240: **fits** (~80% train mem) with the teacher-optimizer-free change.

## Next steps

- Launch the codistill chain (command above) once the RL baseline has a few steps logged.
- Decide run length / stopping point (target: multi-day, many epochs over the 512 problems).
- Optional: add eval cadence (`num_eval_examples=128` is set) to track held-out accuracy for the
  baseline-vs-codistill curves.
- Optional: lower `--mem` to allow concurrent RL + codistill on Mila.
