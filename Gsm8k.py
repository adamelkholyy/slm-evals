from datasets import Dataset, load_dataset

from parser import split


class Gsm8k:

    def load_dataset(self) -> Dataset:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        return ds.map(self.format_gsm8k, remove_columns=ds.column_names)

    def load_dataset_split(self, split_name: str) -> Dataset:
        ds = load_dataset("openai/gsm8k", "main", split=split_name)
        return ds.map(self.format_gsm8k, remove_columns=ds.column_names)

    def load_val_split(self) -> Dataset:
        return self.load_dataset_split("test")

    @staticmethod
    def format_gsm8k(x) -> dict:
        # Keep the dataset as a single "text" field for SFT-style trainers.
        return {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"}

    @staticmethod
    def split_prompt_answer(text: str):
        return split(text)

    def get_negative_example(self, x, answer: str) -> str:
        # Placeholder negative example for DPO/KTO.
        # (Intentionally dumb: easy to replace later.)
        return answer[::-1][:80]

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

        return ds.map(mk, remove_columns=ds.column_names)

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

        return ds.map(mk, remove_columns=ds.column_names)

    def convert_to_kto(self, ds: Dataset) -> Dataset:
        """Convert into KTO format (prompt, completion, label)."""
        rows = []
        for x in ds:
            prompt, answer = self.split_prompt_answer(x["text"])
            rows.append({"prompt": prompt, "completion": answer, "label": True})
            rows.append({"prompt": prompt, "completion": self.get_negative_example(x, answer), "label": False})
        return Dataset.from_list(rows)

