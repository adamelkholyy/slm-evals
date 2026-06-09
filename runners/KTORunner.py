import re

from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import KTOConfig, KTOTrainer

from runners.PostTrainer import PostTrainer
from settings import COMMON, system_prompt
from utils import save_model, split_prompt_answer


class KTORunner(PostTrainer):

    def run(self, model, tokenizer, args):
        ds = self.load_gsm8k()
        ds = self.convert_to_kto(ds)

        # get frozen reference model
        ref_model = AutoModelForCausalLM.from_pretrained(args.model)
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

    # TODO: Check, refactor, re-comment
    def convert_to_kto(self, ds: Dataset) -> Dataset:
        """Convert into KTO format (prompt, completion, label).

        Keep prompt formatting consistent with SFT/GRPO:
        <system_prompt>\n\nQuestion: ...\nAnswer:

        Completions are the GSM8K answer strings (which include the '#### <final>' marker).
        """
        rows = []
        for x in ds:
            prompt_raw, answer = split_prompt_answer(x["text"])

            prompt = f"{system_prompt}\n\n{prompt_raw.strip()}\nAnswer:"

            # Ensure the completion begins with a space (nice tokenization; matches SFT path).
            completion_pos = answer if answer.startswith(" ") else f" {answer}"
            completion_neg = self.get_negative_example(completion_pos)

            rows.append({"prompt": prompt, "completion": completion_pos, "label": True})
            rows.append(
                {"prompt": prompt, "completion": completion_neg, "label": False}
            )
        return Dataset.from_list(rows)

    @staticmethod
    def get_negative_example(answer: str) -> str:
        """Create a well-formed but incorrect completion.

        We keep the gold reasoning text but perturb the final GSM8K marker:
        '#### <answer>' -> '#### <answer+1>' (or a numeric +1 for floats).

        This yields a harder/realistic negative than string reversal.
        """

        # Prefer the last occurrence to avoid cases where the model gives an answer then keeps talking.
        hash_re = re.compile(r"(####\s*)([-+]?\d[\d,\.]*)")
        matches = list(hash_re.finditer(answer))
        if matches:
            m = matches[-1]
            num_str = m.group(2)
            clean = num_str.replace(",", "").strip()
            try:
                if re.fullmatch(r"[-+]?\d+", clean):
                    wrong = str(int(clean) + 1)
                else:
                    wrong = str(float(clean) + 1.0)
            except ValueError:
                wrong = "0"

            start, end = m.span(2)
            return answer[:start] + wrong + answer[end:]

        # Fallback: try changing the last number anywhere.
        any_num_re = re.compile(r"[-+]?\d[\d,\.]*(?!.*[-+]?\d)")
        m2 = any_num_re.search(answer)
        if m2:
            num_str = m2.group(0)
            clean = num_str.replace(",", "").strip()
            try:
                if re.fullmatch(r"[-+]?\d+", clean):
                    wrong = str(int(clean) + 1)
                else:
                    wrong = str(float(clean) + 1.0)
                return answer[: m2.start()] + wrong + answer[m2.end() :]
            except ValueError:
                pass

        # Last-resort fallback: append a clearly wrong final line.
        return answer.rstrip() + "\n#### 0"
