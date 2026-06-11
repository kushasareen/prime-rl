"""Per-micro-batch training primitives for co-distillation.

These re-sequence the RL trainer's per-micro-batch body (``trainer/rl/train.py``)
parameterized over (model, training_mode), so one stage can step the student (rl/opd)
and the teacher (sft) from the same packed batch. Kept deliberately simple: no
context parallelism, multimodal, LoRA, or MoE (asserted off in ``train.py``).
"""

from collections import defaultdict

import torch
import torch.distributed as dist
from torch import nn
from torchtitan.distributed.utils import clip_grad_norm_

from prime_rl.trainer.model import forward
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.data import TensorMicroBatch
from prime_rl.trainer.rl.loss import (
    compute_entropy,
    compute_loss,
    selective_log_softmax,
    shift_tensor_left,
    shift_tensor_right,
)
from prime_rl.trainer.utils import get_response_lengths


def _vocab_size(model: nn.Module) -> int:
    return getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size


def forward_logprobs(model: nn.Module, micro_batch: TensorMicroBatch) -> torch.Tensor:
    """Forward ``model`` over a packed micro-batch and return per-token logprobs ``[1, seq]``.

    Mirrors ``trainer/rl/train.py`` lines 438-472 (no-CP path): forward with per-token
    temperatures, fall back to ``selective_log_softmax`` when the model returns logits,
    then right-shift so logprobs align with the token that produced them.
    """
    input_ids = micro_batch["input_ids"].to("cuda")
    position_ids = micro_batch["position_ids"].to("cuda")
    temperatures = micro_batch["temperatures"].to("cuda")
    labels = shift_tensor_left(input_ids)

    out = forward(model, input_ids, position_ids, labels=labels, temperature=temperatures)

    if out.get("logprobs") is None:
        # VanillaOutputLinear path: compute logprobs from logits with per-token temps.
        scaled_logits = out["logits"] / temperatures.unsqueeze(-1)
        out["logprobs"] = selective_log_softmax(scaled_logits, labels)
        out["entropy"] = compute_entropy(scaled_logits)

    vocab_size = _vocab_size(model)
    logprobs = shift_tensor_right(out["logprobs"], pad_value=torch.log(torch.tensor(1.0 / vocab_size)).item())
    return logprobs


def forward_backward(
    model: nn.Module,
    micro_batches: list[TensorMicroBatch],
    loss_fns: dict,
    training_mode: str,
    parallel_dims: ParallelDims,
    *,
    teacher_logprobs_per_mb: list[torch.Tensor] | None = None,
    max_norm: float | None = None,
) -> dict[str, float]:
    """One forward+backward pass over the batch for ``model`` in ``training_mode``.

    Accumulates gradients across micro-batches (caller does ``optimizer.step()``).
    The loss is normalized by the global (dp_cp) trainable-token count so the gradient
    is the true per-token mean; FSDP's per-rank average is then undone, matching the
    RL trainer. Returns scalar metrics for logging.
    """
    # Global trainable-token denominator (recomputed every call — masks differ per phase).
    local_loss_scale = sum(mb["loss_mask"].sum().item() for mb in micro_batches)
    global_loss_scale = torch.tensor(local_loss_scale, dtype=torch.int64, device="cuda")
    dist.all_reduce(global_loss_scale, op=dist.ReduceOp.SUM, group=parallel_dims.get_mesh("dp_cp").get_group())
    loss_scale = max(global_loss_scale.item(), 1)

    agg: dict[str, list[torch.Tensor]] = defaultdict(list)
    for i, micro_batch in enumerate(micro_batches):
        position_ids = micro_batch["position_ids"].to("cuda")
        response_lengths = get_response_lengths(position_ids)

        trainer_logprobs = forward_logprobs(model, micro_batch)

        teacher_logprobs = None
        if teacher_logprobs_per_mb is not None:
            teacher_logprobs = teacher_logprobs_per_mb[i].to("cuda").squeeze().split(response_lengths)

        loss, loss_metrics = compute_loss(
            trainer_logprobs=trainer_logprobs.squeeze().split(response_lengths),
            inference_logprobs=micro_batch["inference_logprobs"].to("cuda").squeeze().split(response_lengths),
            teacher_logprobs=teacher_logprobs,
            advantages=micro_batch["advantages"].to("cuda").squeeze().split(response_lengths),
            loss_mask=micro_batch["loss_mask"].to("cuda").squeeze().split(response_lengths),
            loss_fns=loss_fns,
            loss_scale=loss_scale,
            training_mode=training_mode,
        )
        loss.backward()

        agg["loss"].append(loss.detach())
        for key, value in loss_metrics.items():
            agg[key].append(value.detach())

    # compute_loss already divided by the global token count; undo FSDP's per-rank average.
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(parallel_dims.fsdp_gradient_divide_factor)

    # compute_loss metrics are per-sequence tensors of varying length across micro-batches
    # (different packed-sequence counts); flatten + concat before reducing. "loss" is a scalar.
    metrics = {key: torch.cat([v.float().reshape(-1) for v in values]).mean().item() for key, values in agg.items()}
    if max_norm is not None:
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm, ep_enabled=parallel_dims.ep_enabled)
        metrics["grad_norm"] = float(grad_norm)
    return metrics


def correct_sft_micro_batches(
    micro_batches: list[TensorMicroBatch], threshold: float
) -> list[TensorMicroBatch]:
    """Build SFT (RFT) micro-batches: keep the trainable mask only on correct samples.

    ``rewards`` is per-token (a sample's scalar reward broadcast to its tokens), so the
    filter is elementwise — shapes are unchanged, keeping per-rank micro-batch counts
    identical across ranks (no FSDP all-gather hang). NaN rewards compare False and are
    excluded automatically.
    """
    out: list[TensorMicroBatch] = []
    for micro_batch in micro_batches:
        if micro_batch["rewards"] is None:
            raise ValueError("RFT requires per-sample rewards in the batch; none present (fake data?).")
        correct = micro_batch["rewards"] >= threshold
        sft_mb = dict(micro_batch)
        sft_mb["loss_mask"] = micro_batch["loss_mask"] & correct
        sft_mb["training_mode"] = "sft"
        out.append(sft_mb)  # type: ignore[arg-type]
    return out


def batch_reward_stats(
    micro_batches: list[TensorMicroBatch], parallel_dims: ParallelDims
) -> tuple[float, float]:
    """Global (mean reward over trainable tokens, max reward) across all ranks."""
    group = parallel_dims.get_mesh("dp_cp").get_group()
    local_sum = torch.zeros((), device="cuda")
    local_count = torch.zeros((), device="cuda")
    local_max = torch.full((), float("-inf"), device="cuda")
    for micro_batch in micro_batches:
        if micro_batch["rewards"] is None:
            continue
        mask = micro_batch["loss_mask"].to("cuda")
        rewards = micro_batch["rewards"].to("cuda")
        masked = rewards[mask]
        masked = masked[~torch.isnan(masked)]
        if masked.numel():
            local_sum = local_sum + masked.sum()
            local_count = local_count + masked.numel()
            local_max = torch.maximum(local_max, masked.max())
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=group)
    mean = (local_sum / local_count).item() if local_count.item() > 0 else float("nan")
    return mean, local_max.item()


def correct_fraction(
    micro_batches: list[TensorMicroBatch], threshold: float, parallel_dims: ParallelDims
) -> float:
    """Fraction of trainable tokens whose sample reward is >= threshold, across ranks."""
    group = parallel_dims.get_mesh("dp_cp").get_group()
    local_correct = torch.zeros((), device="cuda")
    local_total = torch.zeros((), device="cuda")
    for micro_batch in micro_batches:
        mask = micro_batch["loss_mask"].to("cuda")
        local_total = local_total + mask.sum()
        if micro_batch["rewards"] is not None:
            correct = (micro_batch["rewards"].to("cuda") >= threshold) & mask
            local_correct = local_correct + correct.sum()
    dist.all_reduce(local_correct, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(local_total, op=dist.ReduceOp.SUM, group=group)
    return (local_correct / local_total).item() if local_total.item() > 0 else 0.0
