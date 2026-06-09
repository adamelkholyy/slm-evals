import sys
from typing import Any

from settings import DEBUG_EVERY, DEBUG_N

# TODO: REFACTOR! Including logging instead of eprint

def is_rank0() -> bool:
    """Best-effort 'rank0 only' gate for torchrun/accelerate."""
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank() == 0
    except ImportError:
        pass
    return True


_LAST_DEBUG_STEP: int | None = None


def _eprint(*args, **kwargs):
    """Print to stderr so output lands in the SLURM .err file."""
    print(*args, file=sys.stderr, **kwargs)


def maybe_debug_print_grpo(
    *,
    trainer_state: Any,
    prompts: list[Any],
    responses: list[str],
    answers: list[Any],
    extracted: list[Any] | None = None,
    scores: list[Any] | None = None,
    header: str = "GRPO DEBUG",
) -> None:
    """Periodically print samples to stderr (rank0 only)."""

    global _LAST_DEBUG_STEP

    if not is_rank0():
        return

    if trainer_state is None or not hasattr(trainer_state, "global_step"):
        return

    step = int(trainer_state.global_step)
    if DEBUG_EVERY <= 0 or (step % DEBUG_EVERY) != 0:
        return

    if _LAST_DEBUG_STEP == step:
        return
    _LAST_DEBUG_STEP = step

    n = max(1, DEBUG_N)

    _eprint("\n" + "=" * 100)
    _eprint(
        f"[{header}] global_step={step} (printing {min(n, len(responses))} sample(s))"
    )

    for i in range(min(n, len(responses))):
        p = prompts[i]
        r = responses[i]
        a = answers[i]
        g = None if extracted is None else extracted[i]
        sc = None if scores is None else scores[i]

        _eprint("\n[PROMPT]\n" + str(p))
        _eprint("\n[COMPLETION]\n" + str(r)),
        _eprint(f"\n[GT_ANSWER] {a!r}")
        if g is not None:
            _eprint(f"[EXTRACTED_GUESS] {g!r}")
        if sc is not None:
            _eprint(f"[REWARD] {sc}")

    _eprint("=" * 100 + "\n")
