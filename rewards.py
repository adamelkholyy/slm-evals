import re

from utils import get_completion_text, maybe_debug_print_grpo, maybe_log_extra_grpo

# System prompt — kept minimal so the instruct model uses its own reasoning style.
# \boxed{} is a format these models already know from math pre-training data.
system_prompt = (
    "You are a helpful assistant. Solve the following math problem step by step. "
    "Provide your final numerical answer inside \\boxed{}."
)

# Regex to extract the content of \boxed{...}  (last occurrence wins)
match_boxed = re.compile(r"\\boxed\{([^}]*)\}")

# Fallback: grab the last number anywhere in the response
match_last_number = re.compile(r"([\d][\d,.]*)\s*$", re.MULTILINE)

# Reward Function 1: Format Compliance (\boxed{} present?)
def match_format_exactly(completions, **kwargs):
    """
    Reward for including \\boxed{} in the response.
    2.0 if present, 0.0 otherwise.
    """
    scores = []
    for completion in completions:
        response = get_completion_text(completion)
        score = 2.0 if match_boxed.search(response) is not None else 0.0
        scores.append(score)
    return scores


# Reward Function 2: Mathematical Accuracy (extract from \boxed{})
def check_answer_correctness(prompts, completions, answer, **kwargs):
    """
    Graduated scoring for mathematical accuracy:
    - 3.0: Exact match
    - 1.5: Within 10% (close answer)
    - 0.5: Within 20% (reasonable attempt)
    - -0.5: Wrong answer (penalty for incorrect math)
    """
    responses = [get_completion_text(completion) for completion in completions]

    def _extract_boxed(r: str) -> str | None:
        matches = match_boxed.findall(r)
        return matches[-1].strip() if matches else None  # last \boxed{} wins

    extracted_responses = [_extract_boxed(r) for r in responses]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None or true_answer is None:  # No extractable answer / no GT
            scores.append(0)
            continue
            
        # Normalise: strip commas and whitespace
        guess_clean = guess.replace(",", "").strip()
        true_clean = str(true_answer).replace(",", "").strip()

        # Exact string match gets full points
        if guess_clean == true_clean:
            scores.append(3.0)
        else:
            # Try numerical comparison for partial credit
            try:
                ratio = float(guess_clean) / float(true_clean)
                if 0.9 <= ratio <= 1.1:      # Within 10%
                    scores.append(1.5)
                elif 0.8 <= ratio <= 1.2:    # Within 20%
                    scores.append(0.5)
                else:                         # Wrong answer
                    scores.append(-0.5)
            except (ValueError, ZeroDivisionError):
                scores.append(-0.5)           # Invalid numerical format

    # Logging to TRL completions table + periodic stderr printing.
    maybe_log_extra_grpo(
        log_extra=kwargs.get("log_extra"),
        gt_answers=answer,
        extracted=extracted_responses,
    )

    maybe_debug_print_grpo(
        trainer_state=kwargs.get("trainer_state"),
        prompts=prompts,
        responses=responses,
        answers=answer,
        questions=kwargs.get("question"),
        extracted=extracted_responses,
        scores=scores,
        header="GRPO correctness debug",
    )

    return scores


# Reward Function 3: Fallback Number Extraction
def check_numbers_extraction(prompts, completions, answer, **kwargs):
    """
    Fallback extractor: finds the last number in the response (ignoring \\boxed{}).
    Gives partial credit (1.5) if it matches the gold answer, even when the model
    forgot to use \\boxed{}.
    """
    responses = [get_completion_text(completion) for completion in completions]

    def _extract_last_number(r: str) -> str | None:
        matches = match_last_number.findall(r)
        return matches[-1].replace(",", "").strip() if matches else None

    extracted_responses = [_extract_last_number(r) for r in responses]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None or true_answer is None:  # No extractable number / no GT
            scores.append(0)
            continue
            
        try:
            true_val = float(str(true_answer).replace(",", "").strip())
            guess_val = float(guess)
            # Binary scoring: correct (1.5) or incorrect (0)
            scores.append(1.5 if guess_val == true_val else 0.0)
        except (ValueError, TypeError):
            scores.append(0)  # Invalid number format
    
    return scores