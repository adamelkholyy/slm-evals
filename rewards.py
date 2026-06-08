import re

from utils import get_completion_text, maybe_debug_print_grpo, maybe_log_extra_grpo

# Define structured output format for mathematical reasoning
reasoning_start = "<start_working_out>"   # Begin reasoning section
reasoning_end = "<end_working_out>"       # End reasoning section
solution_start = "<SOLUTION>"            # Begin final answer
solution_end = "</SOLUTION>"              # End final answer


# System prompt that teaches the model our desired reasoning structure
system_prompt = f"""You are a mathematical reasoning assistant.
When given a math problem:
1. Show your step-by-step work between {reasoning_start} and {reasoning_end}
2. Provide your final numerical answer between {solution_start} and {solution_end}
3. Be precise and show all calculation steps clearly."""

# Compiled regex patterns for efficient reward computation
match_format = re.compile(
    rf"^[\s]{{0,}}"                      # Optional whitespace at start
    rf"{reasoning_start}.+?{reasoning_end}.*?"  # Reasoning section (non-greedy)
    rf"{solution_start}(.+?){solution_end}"     # Solution section with capture group
    rf"[\s]{{0,}}$",                     # Optional whitespace at end
    flags=re.MULTILINE | re.DOTALL       # Multi-line matching with . matching newlines
)

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", # Extract numbers from solution section
    flags=re.MULTILINE | re.DOTALL        # Flexible pattern matching
)

# Reward Function 1: Exact Format Compliance
def match_format_exactly(completions, **kwargs):
    """
    High reward (3.0) for perfect format adherence
    Ensures model learns the complete structured output pattern
    """
    scores = []
    for completion in completions:
        response = get_completion_text(completion)
        # Check if response matches complete format pattern
        score = 3.0 if match_format.search(response) is not None else 0.0
        scores.append(score)
    return scores


# Reward Function 2: Partial Format Credit
def match_format_approximately(completions, **kwargs):
    """Partial credit for format elements.

    Note: this is intentionally *mutually exclusive* with `match_format_exactly`.
    If the output matches the full format regex, we give 0.0 here (since the
    exact-format reward already handled it).
    """

    scores = []
    for completion in completions:
        response = get_completion_text(completion)
        
        # If it's an exact match, don't double-count format reward.
        if match_format.search(response) is not None:
            scores.append(0.0)
            continue

        score = 0.0
        # Award +0.5 for correct token count, -0.5 for wrong count
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end) == 1 else -0.5
        score += 0.5 if response.count(solution_start) == 1 else -0.5
        score += 0.5 if response.count(solution_end) == 1 else -0.5
        
        scores.append(score)

    return scores


# Reward Function 3: Mathematical Accuracy
def check_answer_correctness(prompts, completions, answer, **kwargs):
    """
    Graduated scoring for mathematical accuracy:
    - 3.0: Exact match
    - 1.5: Within 10% (close answer)
    - 0.5: Within 20% (reasonable attempt)
    - -0.5: Wrong answer (penalty for incorrect math)
    """
    responses = [get_completion_text(completion) for completion in completions]
    
    def _extract_solution(r: str) -> str | None:
        m = match_format.search(r)
        return m.group(1) if m is not None else None

    extracted_responses = [_extract_solution(r) for r in responses]
    
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None or true_answer is None:  # No extractable answer / no GT
            scores.append(0)
            continue
            
        # Exact string match gets full points
        if guess.strip() == str(true_answer).strip():
            scores.append(3.0)
        else:
            # Try numerical comparison for partial credit
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:      # Within 10%
                    scores.append(1.5)
                elif 0.8 <= ratio <= 1.2:    # Within 20%
                    scores.append(0.5)
                else:                         # Wrong answer
                    scores.append(-0.5)
            except (ValueError, ZeroDivisionError):
                scores.append(-0.5)           # Invalid numerical format
    
    # Optional logging to TRL completions table + periodic stdout printing.
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


# Reward Function 4: Number Extraction Ability  
def check_numbers_extraction(prompts, completions, answer, **kwargs):
    """
    Tests the model's ability to extract numerical values from solution sections
    Complementary to exact format matching - focuses on parsing capability
    """
    responses = [get_completion_text(completion) for completion in completions]
    
    def _extract_number(r: str) -> str | None:
        m = match_numbers.search(r)
        return m.group(1) if m is not None else None

    extracted_responses = [_extract_number(r) for r in responses]
    
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None or true_answer is None:  # No extractable number / no GT
            scores.append(0)
            continue
            
        try:
            # Simple numerical equality check
            true_val = float(str(true_answer).strip())
            guess_val = float(guess.strip())
            # Binary scoring: correct (1.5) or incorrect (0)
            scores.append(1.5 if guess_val == true_val else 0.0)
        except (ValueError, TypeError):
            scores.append(0)  # Invalid number format
    
    return scores