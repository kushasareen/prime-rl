"""CPU sanity checks for the co-distillation primitives.

Covers the pure logic that does not need a GPU or distributed init: the in-memory
student->teacher clone, the RFT correct-filter, and the config wiring. The full
stage loop (forward/backward, reward all-reduces) is exercised by the smoke run.
"""

import math

import pytest
import torch
from torch import nn

from prime_rl.configs.codistill import CoDistillTrainerConfig
from prime_rl.trainer.codistill.clone import clone_into_teacher, weight_l2_distance
from prime_rl.trainer.codistill.loop import correct_sft_micro_batches


def _two_models() -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(0)
    student = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    teacher = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    return student, teacher


def test_clone_copies_weights_exactly():
    student, teacher = _two_models()
    assert weight_l2_distance(student, teacher) > 0  # start different

    clone_into_teacher(student, teacher)

    assert weight_l2_distance(student, teacher) == pytest.approx(0.0, abs=1e-6)
    for (ks, vs), (kt, vt) in zip(student.state_dict().items(), teacher.state_dict().items()):
        assert ks == kt
        assert torch.equal(vs, vt)


def test_clone_is_a_copy_not_a_view():
    student, teacher = _two_models()
    clone_into_teacher(student, teacher)
    # Mutating the student afterward must not change the teacher.
    with torch.no_grad():
        next(student.parameters()).add_(1.0)
    assert weight_l2_distance(student, teacher) > 0


def test_clone_rejects_mismatched_architectures():
    student = nn.Linear(8, 8)
    teacher = nn.Linear(8, 4)  # different param shapes/keys
    with pytest.raises((ValueError, RuntimeError)):
        clone_into_teacher(student, teacher)


def _micro_batch(rewards: list[float], loss_mask: list[bool]) -> dict:
    return {
        "input_ids": torch.zeros(1, len(rewards), dtype=torch.long),
        "rewards": torch.tensor(rewards, dtype=torch.float).unsqueeze(0),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.bool).unsqueeze(0),
        "training_mode": "rl",
    }


def test_correct_filter_keeps_only_correct_tokens():
    # Two samples packed: first correct (reward 1.0), second wrong (reward 0.0).
    mb = _micro_batch(rewards=[1.0, 1.0, 0.0, 0.0], loss_mask=[True, True, True, True])
    [out] = correct_sft_micro_batches([mb], threshold=1.0)

    assert out["training_mode"] == "sft"
    assert out["loss_mask"].tolist() == [[True, True, False, False]]
    # Original micro-batch is untouched (we build new dicts).
    assert mb["loss_mask"].tolist() == [[True, True, True, True]]


def test_correct_filter_respects_existing_loss_mask():
    # A correct token that was not trainable (prompt) stays masked out.
    mb = _micro_batch(rewards=[1.0, 1.0], loss_mask=[False, True])
    [out] = correct_sft_micro_batches([mb], threshold=1.0)
    assert out["loss_mask"].tolist() == [[False, True]]


def test_correct_filter_excludes_nan_rewards():
    mb = _micro_batch(rewards=[math.nan, 1.0], loss_mask=[True, True])
    [out] = correct_sft_micro_batches([mb], threshold=1.0)
    assert out["loss_mask"].tolist() == [[False, True]]


def test_correct_filter_raises_without_rewards():
    mb = {"rewards": None, "loss_mask": torch.ones(1, 2, dtype=torch.bool), "training_mode": "rl"}
    with pytest.raises(ValueError):
        correct_sft_micro_batches([mb], threshold=1.0)


def test_teacher_model_defaults_to_student_copy():
    config = CoDistillTrainerConfig(model={"name": "some/student-model", "seq_len": 1024})
    assert config.teacher_model is not None
    assert config.teacher_model.name == "some/student-model"
    # A copy, not the same object — so mutating the teacher config can't alias the student.
    assert config.teacher_model is not config.model


def test_step_counts_have_sane_defaults():
    config = CoDistillTrainerConfig()
    assert config.rl_steps == 0
    assert config.rft_steps >= 1
    assert config.opd_steps >= 1
