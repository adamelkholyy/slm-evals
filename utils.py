import re
import sys
import os
import time
from typing import Any, Tuple
from settings import (
    DEBUG_EVERY,
    DEBUG_N,
    DEBUG_PROMPT_CHARS,
    DEBUG_COMPLETION_CHARS,
    DEBUG_SHOW_FULL_PROMPT
)


def resolve_output_dir(cli_args):
    if cli_args.output_dir:
        out = cli_args.output_dir
    else:
        run_name = cli_args.run_name or f"{cli_args.method}_gsm8k"
        out = os.path.join("outputs", run_name)

    if os.path.exists(out):
        out = f"{out}-{int(time.time())}"

    os.makedirs(out, exist_ok=True)
    return out

def save_model(trainer, label):
    is_lora = hasattr(trainer.model, "save_pretrained") and hasattr(trainer.model, "peft_config")
    out_dir = os.path.join(trainer.args.output_dir, f"{'adapter' if is_lora else 'checkpoint'}-{label}")
    trainer.model.save_pretrained(out_dir)
    trainer.processing_class.save_pretrained(out_dir)
    print(f"{'Adapter' if is_lora else 'Model'} saved to {out_dir}")



def split(text: str) -> Tuple[str, str]:

    SPLIT_PATTERN = re.compile(
        r"(\nAnswer:|\nCorrect:|\nSolution\n|\nEndings:\n)"
    )
    matches = list(SPLIT_PATTERN.finditer(text))

    if matches:
        m = matches[-1]  # rightmost match
        split_start = m.start()
        split_end = m.end()

        prompt = text[:split_start]
        answer = text[split_end:]
        return prompt, answer

    # else default to last newline split
    idx = text.rfind("\n")
    if idx != -1:
        return text[:idx + 1], text[idx + 1:]

    return text, ""


def is_rank0() -> bool:
    """Best-effort 'rank0 only' gate for torchrun/accelerate."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
    except ImportError:
        pass
    return True


def truncate(s: str, n: int) -> str:
    s = s.replace("\r", "")
    return s if len(s) <= n else s[:n] + "...<truncated>"


def get_completion_text(completion: Any) -> str:
    """Return generated text for both plain-text and chat-style completions.

    In TRL GRPO:
      - If your dataset has `prompt: str`, completions are returned as plain strings.
      - If your dataset has `prompt: list[{'role','content'}]`, completions are returned as chat messages.
    """

    if isinstance(completion, str):
        return completion

    if isinstance(completion, list) and completion:
        msg0 = completion[0]
        if isinstance(msg0, dict) and "content" in msg0:
            return str(msg0["content"])

    if isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])

    return str(completion)




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
    questions: list[Any] | None = None,
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
    _eprint(f"[{header}] global_step={step} (printing {min(n, len(responses))} sample(s))")

    for i in range(min(n, len(responses))):
        q = None if questions is None else questions[i]
        p = prompts[i]
        r = responses[i]
        a = answers[i]
        g = None if extracted is None else extracted[i]
        sc = None if scores is None else scores[i]

        if q is not None:
            _eprint("\n[QUESTION]\n" + truncate(str(q), 800))

        if DEBUG_SHOW_FULL_PROMPT:
            _eprint("\n[PROMPT]\n" + str(p))
        else:
            _eprint("\n[PROMPT]\n" + truncate(str(p), DEBUG_PROMPT_CHARS))

        _eprint("\n[COMPLETION]\n" + truncate(str(r), DEBUG_COMPLETION_CHARS))
        _eprint(f"\n[GT_ANSWER] {a!r}")
        if g is not None:
            _eprint(f"[EXTRACTED_GUESS] {g!r}")
        if sc is not None:
            _eprint(f"[REWARD] {sc}")

    _eprint("=" * 100 + "\n")


def maybe_log_extra_grpo(*, log_extra: Any, gt_answers: list[Any], extracted: list[Any]) -> None:
    """Add GT/extracted fields into TRL's completions table.

    Requires GRPOConfig.log_completions=True.
    """

    if not callable(log_extra):
        return

    try:
        log_extra("gt_answer", [str(a) for a in gt_answers])
        log_extra("extracted_guess", ["" if g is None else str(g) for g in extracted])
    except Exception:
        # Never crash training due to debugging.
        return
