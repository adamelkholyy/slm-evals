import re

from datasets import Dataset, load_dataset
from rewards import system_prompt
from utils import split


class Gsm8k:

    def load_raw_dataset_split(self, split_name: str) -> Dataset:
        """Raw GSM8K rows with 'question'/'answer' columns."""
        return load_dataset("openai/gsm8k", "main", split=split_name)

    def load_dataset(self) -> Dataset:
        """Default GSM8K formatted as a single `text` column.

        Used by DPO/KTO/Reward in this codebase.
        """
        ds = self.load_raw_dataset_split("train")
        cols_to_remove = [c for c in ds.column_names if c != "text"]
        return ds.map(self.format_gsm8k, remove_columns=cols_to_remove, load_from_cache_file=False)

    def load_sft_dataset(self) -> Dataset:
        """Prompt/completion dataset for SFT.

        We keep prompt and completion separate so we can train *completion-only*
        with a completion-only data collator.
        """
        ds = self.load_raw_dataset_split("train")
        cols_to_remove = [c for c in ds.column_names if c not in ("prompt", "completion")]
        return ds.map(self.format_gsm8k_sft, remove_columns=cols_to_remove, load_from_cache_file=False)

    def load_grpo_dataset(self) -> Dataset:
        """Raw GSM8K rows for GRPO.

        Important: GRPO needs access to the original GSM8K `answer` string that
        contains the '####' final-answer delimiter.
        """

        return self.load_raw_dataset_split("train")

    def load_dataset_split(self, split_name: str) -> Dataset:
        ds = self.load_raw_dataset_split(split_name)
        cols_to_remove = [c for c in ds.column_names if c != "text"]
        return ds.map(self.format_gsm8k, remove_columns=cols_to_remove, load_from_cache_file=False)

    def load_val_split(self) -> Dataset:
        return self.load_dataset_split("test")

    @staticmethod
    def format_gsm8k(x) -> dict:
        # Single text field (useful for trainers that expect `text`).
        return {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"}

    @staticmethod
    def format_gsm8k_sft(x) -> dict:
        """Format as prompt/completion for completion-only SFT."""
        prompt = f"{system_prompt}\n\nQuestion: {x['question']}\nAnswer:"
        completion = f" {x['answer']}"  # leading space so it tokenises nicely
        return {"prompt": prompt, "completion": completion}

    @staticmethod
    def split_prompt_answer(text: str):
        return split(text)

    def get_negative_example(self, x, answer: str) -> str:
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

    @staticmethod
    def reward_function(completions, prompts=None, **kw):
        # Placeholder reward. For GRPO, return one scalar per completion.
        return [0.0] * len(completions)

    def convert_to_preference(self, ds: Dataset) -> Dataset:
        """Convert a GSM8K dataset with a 'text' column into DPO triples."""

        def mk(x):
            prompt, answer = self.split_prompt_answer(x["text"])
            rejected = self.get_negative_example(x, answer)
            return {"prompt": prompt, "chosen": answer, "rejected": rejected}

        return ds.map(mk, remove_columns=ds.column_names, load_from_cache_file=False)

    def convert_to_reward(self, ds: Dataset, tok) -> Dataset:
        """Convert into RewardTrainer format (tokenized chosen/rejected pairs)."""

        def mk(x):
            prompt, answer = self.split_prompt_answer(x["text"])
            chosen = tok(prompt + answer, truncation=True, max_length=512)
            rejected = tok(prompt + self.get_negative_example(x, answer), truncation=True, max_length=512)
            return {
                "input_ids_chosen": chosen["input_ids"],
                "attention_mask_chosen": chosen["attention_mask"],
                "input_ids_rejected": rejected["input_ids"],
                "attention_mask_rejected": rejected["attention_mask"],
            }

        return ds.map(mk, remove_columns=ds.column_names, load_from_cache_file=False)

    def convert_to_kto(self, ds: Dataset) -> Dataset:
        """Convert into KTO format (prompt, completion, label).

        Keep prompt formatting consistent with SFT/GRPO:
          <system_prompt>\n\nQuestion: ...\nAnswer:

        Completions are the GSM8K answer strings (which include the '#### <final>' marker).
        """
        rows = []
        for x in ds:
            prompt_raw, answer = self.split_prompt_answer(x["text"])

            prompt = f"{system_prompt}\n\n{prompt_raw.strip()}\nAnswer:"

            # Ensure the completion begins with a space (nice tokenization; matches SFT path).
            completion_pos = answer if answer.startswith(" ") else f" {answer}"
            completion_neg = self.get_negative_example(x, completion_pos)

            rows.append({"prompt": prompt, "completion": completion_pos, "label": True})
            rows.append({"prompt": prompt, "completion": completion_neg, "label": False})
        return Dataset.from_list(rows)


    def convert_to_grpo(self, ds: Dataset) -> Dataset:
        ds = ds.map(self.grpo_processing, remove_columns=ds.column_names, load_from_cache_file=False)

        # Drop rows with no extractable gold final answer.
        before_count = len(ds)
        ds = ds.filter(lambda x: x["answer"] is not None, load_from_cache_file=False)
        dropped = before_count - len(ds)
        if dropped:
            print(f"[GSM8K] Warning: dropped {dropped} rows with answer=None (failed '####' extraction)")

        return ds

    @staticmethod
    def grpo_processing(x):
        """Convert GSM8K example to GRPO format."""

        def extract_hash_answer(text):
            """Extract numerical answer from GSM8K format (#### marker)"""
            if "####" not in text:
                return None
            # GSM8K uses format: "Explanation... #### 42"
            return text.split("####")[1].strip()


        if "question" in x and "answer" in x:
            question = x["question"]
            answer_text = x["answer"]
        elif "text" in x:
            # If we accidentally pass the formatted dataset (with only 'text'),
            # recover the original fields via the splitter.
            prompt_text, ans = split(x["text"])
            question = prompt_text.replace("Question:", "").strip()
            answer_text = ans
        else:
            raise KeyError(f"GSM8K GRPO processing expected keys 'question'/'answer' or 'text', got: {list(x.keys())}")

        answer = extract_hash_answer(answer_text)

        # Plain-text prompt (no chat template)
        prompt = f"{system_prompt}\n\n{question}\n"

        return {
            "prompt": prompt,
            "question": question,  # for debugging/logging
            "answer": answer,  # gold final answer for reward functions
        }

