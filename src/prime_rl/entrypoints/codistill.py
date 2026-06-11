from prime_rl.configs.codistill import CoDistillConfig
from prime_rl.entrypoints.rl import rl
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title


def main():
    """Launcher for staged student/teacher co-distillation (`uv run codistill`).

    Reuses the rl launcher (inference + orchestrator + trainer supervision) but points
    the trainer at the co-distillation loop. The orchestrator runs plain rl generation;
    the teacher/RFT/OPD logic lives in the trainer.
    """
    set_proc_title("Launcher")
    config = cli(CoDistillConfig)
    if config.slurm is not None:
        raise ValueError(
            "codistill does not support the prime-rl SLURM integration yet — submit scripts/mila/codistill.sh instead."
        )
    rl(config, trainer_module="prime_rl.trainer.codistill.train")


if __name__ == "__main__":
    main()
