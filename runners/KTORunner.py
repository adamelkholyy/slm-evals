import re

from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import KTOConfig, KTOTrainer

from runners.PostTrainer import PostTrainer
from settings import COMMON, system_prompt
from utils import save_model, split_prompt_answer

# Matches a GSM8K final-answer marker and the number that follows it.
_HASH_RE = re.compile(r"(####\s*)([-+]?\d[\d,\.]*)")

# Matches any number (used as fallback when no #### marker is present).
_NUM_RE = re.compile(r"[-+]?\d[\d,\.]*")


class KTORunner(PostTrainer):

    def run(self, model, tokenizer, args):
        ds = self.load_gsm8k()
        ds = self.convert_to_kto(ds)

        # The reference model must stay frozen and match the training model's
        # dtype/device layout so the KL term is computed consistently.
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=model.dtype,
            device_map=model.hf_device_map if hasattr(model, "hf_device_map") else "auto",
        )
        for param in ref_model.parameters():
            param.requires_grad = False

        config = dict(
            COMMON,
            output_dir=args.output_dir,
            remove_unused_columns=False,
            beta=0.1,
        )

        trainer = KTOTrainer(
            model=model,
            train_dataset=ds,
            ref_model=ref_model,
            processing_class=tokenizer,
            args=KTOConfig(**config),
        )
        trainer.train()
        save_model(trainer, "kto")

    def convert_to_kto(self, ds: Dataset) -> Dataset:
        """Convert GSM8K examples into KTO format (prompt, completion, label).

        Each source example produces two rows:
          - label=True  with the gold completion
          - label=False with a perturbed completion (correct reasoning, wrong answer)

        Prompt format matches SFT/GRPO:
            <system_prompt>\\n\\nQuestion: ...\\nAnswer:
        """
        rows = []
        for x in ds:
            prompt_raw, answer = split_prompt_answer(x["text"])
            prompt = f"{system_prompt}\n\n{prompt_raw.strip()}\nAnswer:"

            # Leading space for clean tokenization boundary (matches SFT path).
            completion_pos = answer if answer.startswith(" ") else f" {answer}"
            completion_neg = self._get_negative_example(completion_pos)

            rows.append({"prompt": prompt, "completion": completion_pos, "label": True})
            rows.append({"prompt": prompt, "completion": completion_neg, "label": False})

        return Dataset.from_list(rows)

    @staticmethod
    def _get_negative_example(answer: str) -> str:
        """Return a plausible but incorrect completion for KTO negative training.

        Strategy: preserve the full reasoning chain and perturb only the final
        numeric answer by +1. This produces harder negatives than random noise
        while keeping the text otherwise valid.

        Perturbation priority:
          1. Last '#### <n>' marker (canonical GSM8K format).
          2. Last bare number anywhere in the string (rare fallback).
          3. Append '#### 0' when no number is found at all.
        """

        hash_matches = list(_HASH_RE.finditer(answer))
        if hash_matches:
            m = hash_matches[-1]
            wrong = _perturb_number(m.group(2))
            return answer[: m.start(2)] + wrong + answer[m.end(2) :]

        num_matches = list(_NUM_RE.finditer(answer))
        if num_matches:
            m = num_matches[-1]
            wrong = _perturb_number(m.group(0))
            return answer[: m.start()] + wrong + answer[m.end() :]

        return answer.rstrip() + "\n#### 0"


def _perturb_number(num_str: str) -> str:
    """Increment a numeric string by 1, preserving integer vs float type.

    Commas are stripped for parsing; the result is a plain decimal string.
    Returns '0' on parse error.

    Examples:
        '42'     -> '43'
        '3.5'    -> '4.5'
        '1,000'  -> '1001'
    """
    clean = num_str.replace(",", "")
    try:
        if re.fullmatch(r"[-+]?\d+", clean):
            return str(int(clean) + 1)
        return str(float(clean) + 1.0)
    except ValueError:
        return "0"