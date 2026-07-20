from pydantic import Field, model_validator

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.trainer import AdamWConfig, ModelConfig, OptimizerConfig, TrainerConfig


class CoDistillTrainerConfig(TrainerConfig):
    """Trainer config for staged student/teacher co-distillation.

    Each stage reuses one batch of student generations: optionally ``rl_steps``
    GRPO steps on the student, clone the student into the teacher (in memory),
    ``rft_steps`` SFT steps on the teacher over the student's *correct* generations,
    then ``opd_steps`` on-policy-distillation steps on the student from the teacher.
    The teacher lives in this process; ``model`` is the student, ``teacher_model``
    the teacher (defaults to a copy of the student).
    """

    teacher_model: ModelConfig | None = None
    """Teacher model. Defaults to a copy of ``model`` (the teacher is re-cloned from the student each stage anyway)."""

    teacher_optim: OptimizerConfig = AdamWConfig()
    """Optimizer for the teacher's per-stage RFT. Re-initialized each stage (the teacher is re-cloned)."""

    rl_steps: int = Field(0, ge=0)
    """R: optional GRPO steps on the student before cloning. 0 disables the RL step."""

    rft_steps: int = Field(1, ge=1)
    """E: SFT steps on the teacher over the correct subset of the stage batch."""

    opd_steps: int = Field(1, ge=1)
    """D: on-policy-distillation steps on the student from the teacher."""

    correct_reward_threshold: float | None = None
    """Reward at/above which a sample is "correct" for RFT. None -> the stage batch's max reward."""

    @property
    def max_student_optimizer_steps(self) -> int | None:
        """Total student optimizer updates across all CoDistill stages (rl + opd steps per stage)."""
        if self.max_steps is None:
            return None
        return self.max_steps * (self.rl_steps + self.opd_steps)

    @model_validator(mode="after")
    def default_teacher_model(self):
        if self.teacher_model is None:
            self.teacher_model = self.model.model_copy(deep=True)
        return self


class CoDistillConfig(RLConfig):
    """Top-level launcher config for ``uv run codistill``.

    Mirrors ``RLConfig`` (reusing its shared-field propagation, deployment, and
    weight-broadcast machinery) but swaps in the co-distillation trainer. The
    orchestrator stays in plain ``rl`` generation mode — the teacher/RFT/OPD logic
    is entirely trainer-side, so no ``orchestrator.teacher`` or second server.
    """

    trainer: CoDistillTrainerConfig

    @model_validator(mode="after")
    def force_rl_generation(self):
        if self.orchestrator.training_mode != "rl":
            raise ValueError(
                "codistill drives generation in plain 'rl' mode (teacher/RFT/OPD are trainer-side). "
                "Do not set orchestrator.training_mode."
            )
        return self
