import torch
from torch import nn
from torch.distributed.tensor import DTensor


@torch.no_grad()
def clone_into_teacher(student: nn.Module, teacher: nn.Module) -> None:
    """Copy the student's weights into the teacher, in memory (no disk).

    Both models are FSDP2-sharded over the same mesh with identical placements
    (same architecture, same parallel dims), so each parameter is a shard-local
    copy — no gather, no checkpoint round-trip. Mirrors the ``to_local()`` idiom
    used in ``trainer/weights.py`` and ``trainer/optim.py``.
    """
    student_sd = student.state_dict()
    teacher_sd = teacher.state_dict()
    if student_sd.keys() != teacher_sd.keys():
        raise ValueError("student and teacher state_dicts differ in keys; they must share an architecture")
    for key, s_val in student_sd.items():
        t_val = teacher_sd[key]
        if isinstance(s_val, DTensor):
            if s_val.placements != t_val.placements or s_val.device_mesh != t_val.device_mesh:
                raise ValueError(
                    f"DTensor placement/mesh mismatch for '{key}'; student and teacher must share one mesh"
                )
            t_val.to_local().copy_(s_val.to_local())
        else:
            t_val.copy_(s_val)


@torch.no_grad()
def weight_l2_distance(model_a: nn.Module, model_b: nn.Module) -> float:
    """Global L2 distance between two same-architecture models (sanity metric).

    Sums local shard contributions; for a single-rank run this is the full norm.
    """
    total = torch.zeros((), device=next(model_a.parameters()).device)
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    for key, a_val in sd_a.items():
        b_val = sd_b[key]
        a_local = a_val.to_local() if isinstance(a_val, DTensor) else a_val
        b_local = b_val.to_local() if isinstance(b_val, DTensor) else b_val
        total = total + ((a_local.float() - b_local.float()) ** 2).sum()
    return total.sqrt().item()
