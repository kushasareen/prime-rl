# reasoning-core codistill — status & handoff

Status of the first "serious dataset" codistill experiment on Tamia, plus the proxy
bug that blocked it. Written so debugging can continue elsewhere (e.g. Mila, where
allocations come faster) — the run itself is cluster-agnostic; only the proxy bit is
Tamia-specific.

## The experiment

Staged student/teacher co-distillation on **`reasoning-core-env`** (formally-verified
symbolic reasoning, 40 procedural task types), filtered to difficulty **levels 2–3**,
**512 problems** (data-efficiency regime). Binary reward (`score_answer==1`), so
`correct_reward_threshold=0.8` == "fully correct".

- Config (1.5B): `configs/debug/codistill_reasoning_core.toml` — DeepSeek-R1-Distill-Qwen-1.5B, 2 infer + 2 train.
- Config (4B): `configs/debug/codistill_reasoning_core_qwen3_4b.toml` — Qwen/Qwen3-4B, 1 infer + 3 train (4B×2 + 2 AdamW ≈ 128 GB sharded; 2 train OOMs).
- Dataset prep: `scripts/tamia/prep_reasoning_core.py` → parquet at `$SCRATCH/datasets/reasoning_core_l2-3_n512`.
- Launch: `CONFIG=<cfg> NUM_INFER_GPUS=.. NUM_TRAIN_GPUS=.. sbatch scripts/tamia/codistill.sh`

## The bug (FIXED) — proxy swallowed localhost traffic

Both jobs (1.5B `354382`, 4B `354854`) ran on 2026-06-30 and **died identically ~3 min
in**, during orchestrator setup — NOT training, NOT OOM (the 4B memory split was fine):

```
orchestrator.setup → student_inference.wait_for_ready → maybe_check_has_model
→ httpx .json() on an EMPTY response
→ json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Cause:** the `httpproxy` integration set `http_proxy`/`https_proxy` to Tamia's Squid
proxy so wandb/HF work online on compute nodes — but `no_proxy` only had
`tamia.ecpia.ca`. The orchestrator talks to its *own* vLLM server at
`http://localhost:8000`; that localhost request got routed through Squid, which returns
an empty body → JSON parse blows up. (reverse-text worked earlier only because it ran
before the proxy change.)

**Fix** (`scripts/tamia/env.sh`, commit `02046d9a`): add localhost forms to `no_proxy`
so internal traffic bypasses the proxy while external still goes through it:

```sh
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1,tamia.ecpia.ca"
export NO_PROXY="$no_proxy"
```

**Validated locally** (login node) by reproducing the exact error with httpx: with the
proxy set and no localhost in `no_proxy`, `.json()` on the localhost response raises the
identical `JSONDecodeError`; with localhost in `no_proxy`, it bypasses the proxy and
parses fine.

This matches the fix in the **PipelineRL** repo (`~/PipelineRL/scripts/run.sh`:
`export no_proxy=localhost,127.0.0.1,0.0.0.0,::1`). PipelineRL additionally forces
IPv4 (`pipelinerl/world.py`: use `127.0.0.1`, not `localhost`) "to avoid IPv6/Proxy
issues on Compute Canada" — needed there because of `MASTER_ADDR` resolution. prime-rl's
httpx client matches the `no_proxy` entry by the literal hostname string in the
base_url (`localhost`), so it's covered without the IPv4 pin. **Fallback if a localhost
request still hits the proxy:** pin the inference base_url to IPv4, e.g.
`--orchestrator.student.client.base_url http://127.0.0.1:8000/v1` (+ teacher), and keep
`127.0.0.1` in `no_proxy`.

## Current status (2026-06-30)

- Fix applied + pushed (`exp/codistill` @ `02046d9a`+). Requeued **`359269`** (1.5B) and
  **`359270`** (4B) with the fix (env.sh is sourced at job runtime).
- Tamia queue is brutally oversubscribed (~348 pending GPU jobs vs ~64 running; whole-node
  4-GPU allocations only). Est. starts keep slipping by days. Our account fairshare is
  depressed by heavy account-mate usage (not us). This is why validating here is slow.

## Not yet validated / next steps

- **The reasoning-core codistill run has never gotten past orchestrator setup** — both
  prior attempts died on the proxy bug *before* Step 0. So still unconfirmed:
  1. that the run trains end-to-end (GRPO → teacher RFT → OPD) on this env, and
  2. the **base reward at Step 0** — i.e. whether levels 2–3 land in the low-but-nonzero
     band we want (retune the level filter in `prep_reasoning_core.py` if it's ~0 or high).
- Fastest path: validate on a cluster with quick allocations. The run is portable; only
  `env.sh`'s proxy block is Tamia-specific (Mila compute nodes have direct internet, so
  no proxy needed there).
- Once Step 0 looks healthy, promote from the 10-stage smoke to a longer run.
