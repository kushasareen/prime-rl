import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before model imports

from datetime import timedelta

import torch

from prime_rl.configs.codistill import CoDistillTrainerConfig
from prime_rl.trainer.codistill.clone import clone_into_teacher, weight_l2_distance
from prime_rl.trainer.codistill.loop import (
    batch_reward_stats,
    correct_fraction,
    correct_sft_micro_batches,
    forward_backward,
    forward_logprobs,
)
from prime_rl.trainer.model import setup_model, setup_tokenizer
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.rl.broadcast import setup_weight_broadcast
from prime_rl.trainer.rl.data import DataLoader
from prime_rl.trainer.rl.loss import setup_loss_fns
from prime_rl.trainer.runs import Progress, setup_multi_run_manager
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit


def _assert_supported(config: CoDistillTrainerConfig) -> None:
    """Fail fast on features the co-distillation loop deliberately does not implement."""
    if config.model.cp > 1:
        raise ValueError("codistill does not support context parallelism (model.cp must be 1).")
    if config.model.lora is not None:
        raise ValueError("codistill does not support LoRA.")
    if config.max_concurrent_runs != 1:
        raise ValueError("codistill does not support multi-run training (max_concurrent_runs must be 1).")
    if config.enable_router_replay:
        raise ValueError("codistill does not support router replay / MoE routing.")
    if config.data.fake:
        raise ValueError("codistill needs real rollouts with rewards; fake data has none.")


@clean_exit
def train(config: CoDistillTrainerConfig):
    world = get_world()
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
    logger.info(f"Starting co-distillation trainer in {world} in {config.output_dir}")
    _assert_supported(config)

    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)

    setup_torch_distributed(timeout=timedelta(seconds=config.dist_timeout_seconds))
    torch.set_float32_matmul_precision(config.matmul_precision)
    setup_multi_run_manager(config.output_dir, 1, torch.device("cuda", world.local_rank), None)

    parallel_dims = get_parallel_dims(config.model)

    logger.info(f"Initializing student model ({config.model.name})")
    student = setup_model(config.model, parallel_dims, False)
    logger.info(f"Initializing teacher model ({config.teacher_model.name})")
    teacher = setup_model(config.teacher_model, parallel_dims, False)
    tokenizer = setup_tokenizer(config.tokenizer)

    loss_fns = setup_loss_fns(config.loss)
    student_optimizer = setup_optimizer(
        config.optim, list(student.named_parameters()), parallel_dims, lora=False, cpu_offload=False
    )
    student_scheduler = setup_scheduler(student_optimizer, config.scheduler, config.max_steps, config.optim.lr)

    if config.data.fake:  # unreachable (guarded above) — kept explicit for the reader
        weight_broadcast = None
    else:
        logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
        weight_broadcast = setup_weight_broadcast(config.output_dir, config.weight_broadcast, None)

    dataloader = DataLoader(
        config.output_dir,
        0,
        parallel_dims.get_mesh("dp").size(),
        config.model.seq_len,
        config.model.cp,
        tokenizer,
        config.rollout_transport,
    )

    progress = Progress()
    logger.info(f"Starting co-distillation loop (stages={config.max_steps or 'infinite'})")

    while config.max_steps is None or progress.step < config.max_steps:
        stage = progress.step

        # Ship the previous stage's updated student to the inference server (skip stage 0).
        if weight_broadcast is not None and stage > 0:
            weight_broadcast.broadcast_weights(student, step=stage)
            if config.weight_broadcast.type == "filesystem":
                weight_broadcast.maybe_clean(None)

        # One batch of student rollouts for this whole stage.
        dataloader.wait_for_batch()
        micro_batches = dataloader.get_batch()

        reward_mean, reward_max = batch_reward_stats(micro_batches, parallel_dims)
        threshold = config.correct_reward_threshold if config.correct_reward_threshold is not None else reward_max
        frac_correct = correct_fraction(micro_batches, threshold, parallel_dims)
        metrics: dict[str, float] = {
            "codistill/student_reward_mean": reward_mean,
            "codistill/reward_max": reward_max,
            "codistill/correct_frac": frac_correct,
            "codistill/correct_threshold": threshold,
        }

        # (R) Optional GRPO steps on the student over this stage's batch.
        for _ in range(config.rl_steps):
            rl_metrics = forward_backward(
                student, micro_batches, loss_fns, "rl", parallel_dims, max_norm=config.optim.max_norm
            )
            student_optimizer.step()
            student_optimizer.zero_grad()
            student_scheduler.step()
        if config.rl_steps:
            metrics["codistill/rl_loss"] = rl_metrics["loss"]

        # Clone the (updated) student into the teacher, in memory; reset teacher optimizer.
        clone_into_teacher(student, teacher)
        metrics["codistill/clone_dist_after_clone"] = weight_l2_distance(student, teacher)
        teacher_optimizer = setup_optimizer(
            config.teacher_optim, list(teacher.named_parameters()), parallel_dims, lora=False, cpu_offload=False
        )

        # (E) RFT: SFT the teacher on the student's CORRECT generations.
        teacher.train()
        rft_micro_batches = correct_sft_micro_batches(micro_batches, threshold)
        for _ in range(config.rft_steps):
            rft_metrics = forward_backward(
                teacher, rft_micro_batches, loss_fns, "sft", parallel_dims, max_norm=config.teacher_optim.max_norm
            )
            teacher_optimizer.step()
            teacher_optimizer.zero_grad()
        metrics["codistill/teacher_rft_nll"] = rft_metrics["nll"]
        metrics["codistill/clone_dist_after_rft"] = weight_l2_distance(student, teacher)

        # Teacher is frozen during OPD — compute its logprobs once and cache.
        teacher.eval()
        with torch.no_grad():
            teacher_logprobs = [forward_logprobs(teacher, mb).detach() for mb in micro_batches]

        # (D) OPD: distill the student toward the teacher on the same batch.
        first_opd_kl = last_opd_kl = float("nan")
        for opd_step in range(config.opd_steps):
            opd_metrics = forward_backward(
                student,
                micro_batches,
                loss_fns,
                "opd",
                parallel_dims,
                teacher_logprobs_per_mb=teacher_logprobs,
                max_norm=config.optim.max_norm,
            )
            student_optimizer.step()
            student_optimizer.zero_grad()
            student_scheduler.step()
            if opd_step == 0:
                first_opd_kl = opd_metrics["teacher_kl"]
            last_opd_kl = opd_metrics["teacher_kl"]
        metrics["codistill/opd_teacher_kl_first"] = first_opd_kl
        metrics["codistill/opd_teacher_kl_last"] = last_opd_kl
        metrics["codistill/opd_mismatch_kl"] = opd_metrics["unmasked_mismatch_kl"]

        metrics["codistill/lr"] = student_optimizer.param_groups[0]["lr"]
        metrics["step"] = stage
        logger.success(
            f"Stage {stage} | Reward {reward_mean:.4f} | Correct {frac_correct:.1%} | "
            f"RFT NLL {metrics['codistill/teacher_rft_nll']:.4f} | "
            f"OPD KL {first_opd_kl:.4f}->{last_opd_kl:.4f}"
        )
        monitor.log(metrics, step=stage)

        progress.step += 1

    # Broadcast the final student so the orchestrator can advance to the terminal
    # policy version, assemble its draining batch (step >= max_steps), and tear down
    # cleanly. The rl trainer broadcasts at the top of the terminal step before
    # breaking; our while-condition skips that iteration, so we do it explicitly here.
    if weight_broadcast is not None and config.max_steps is not None:
        weight_broadcast.broadcast_weights(student, step=progress.step)

    logger.success("Co-distillation trainer finished!")


def main():
    """Entry-point for the co-distillation trainer (launched by `uv run codistill` under torchrun)."""
    set_proc_title("Trainer")
    config = cli(CoDistillTrainerConfig)
    train(config)


if __name__ == "__main__":
    main()
