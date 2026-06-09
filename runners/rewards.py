import re

from debug import maybe_debug_print_grpo
from utils import get_completion_text

# Regex to extract the content after a GSM8K-style final marker.
# We prefer the last occurrence to handle "correct answer then hallucinate" cases.
match_hash_any = re.compile(r"####\s*([-+]?\d[\d,\.]*)")
match_hash_endline = re.compile(r"####\s*([-+]?\d[\d,\.]*)\s*$", re.MULTILINE)

# Fallback: grab the last number anywhere in the response
match_last_number = re.compile(r"([\d][\d,.]*)\s*$", re.MULTILINE)


# Reward Function 1: Format Compliance (#### <answer>)
def match_format_exactly(completions, **kwargs):
    """Reward for including a GSM8K-style final marker.

    - 2.0 if a line ends with: #### <number>
    - 1.0 if it contains: #### <number> somewhere
    - 0.0 otherwise
    """
    scores = []
    for completion in completions:
        response = get_completion_text(completion)
        if match_hash_endline.search(response) is not None:
            scores.append(2.0)
        elif match_hash_any.search(response) is not None:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores


def _extract_hash_answer(text: str) -> str | None:
    """Extract the last '#### <answer>' occurrence from a completion."""
    matches = match_hash_any.findall(text)
    return matches[-1].strip() if matches else None


def _clean_number_str(s: str) -> str:
    return s.replace(",", "").strip()


# Reward Function 2: Mathematical Accuracy (extract from ####)
def check_answer_correctness(prompts, completions, answer, **kwargs):
    """
    Graduated scoring for mathematical accuracy:
    - 3.0: Exact match
    - 1.5: Within 10% (close answer)
    - 0.5: Within 20% (reasonable attempt)
    - -0.5: Wrong answer (penalty for incorrect math)
    """
    responses = [get_completion_text(completion) for completion in completions]

    extracted_responses = [_extract_hash_answer(r) for r in responses]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None or true_answer is None:  # No extractable answer / no GT
            scores.append(0)
            continue

        guess_clean = _clean_number_str(guess)
        true_clean = _clean_number_str(str(true_answer))

        # Exact string match gets full points
        if guess_clean == true_clean:
            scores.append(3.0)
        else:
            # Try numerical comparison for partial credit
            try:
                ratio = float(guess_clean) / float(true_clean)
                if 0.9 <= ratio <= 1.1:  # Within 10%
                    scores.append(1.5)
                elif 0.8 <= ratio <= 1.2:  # Within 20%
                    scores.append(0.5)
                else:  # Wrong answer
                    scores.append(-0.5)
            except (ValueError, ZeroDivisionError):
                scores.append(-0.5)  # Invalid numerical format

    maybe_debug_print_grpo(
        trainer_state=kwargs.get("trainer_state"),
        prompts=prompts,
        responses=responses,
        answers=answer,
        extracted=extracted_responses,
        scores=scores,
        header="GRPO correctness debug",
    )

    return scores


# Reward Function 3: Fallback Number Extraction
def check_numbers_extraction(prompts, completions, answer, **kwargs):
    """Fallback extractor: finds the last number in the response.

    Gives partial credit (1.5) if it matches the gold answer, even when the model
    forgot to emit the final marker: #### <answer>.
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
