import re
from typing import Tuple

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


# split_tokens = ["\nAnswer:", "\nCorrect:", "\nSolution\n", "\nEndings:\n"]
# for sep in split_tokens:
#     idx = text.rfind(sep)
#     if idx != -1:
#         return text[: idx + len(sep)], text[idx + len(sep) :]
# # fallback: split at last newline
# idx = text.rfind("\n")
# return (text[: idx + 1], text[idx + 1 :]) if idx != -1 else (text, "")
