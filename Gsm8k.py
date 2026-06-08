from datasets import Dataset, load_dataset
from rewards import system_prompt
from utils import split


class Gsm8k:

    def load_raw_dataset_split(self, split_name: str) -> Dataset:
        """Raw GSM8K rows with 'question'/'answer' columns."""
        return load_dataset("openai/gsm8k", "main", split=split_name)

    def load_dataset(self) -> Dataset:
        ds = self.load_raw_dataset_split("train")
        cols_to_remove = [c for c in ds.column_names if c != "text"]
        return ds.map(self.format_gsm8k, remove_columns=cols_to_remove)

    def load_grpo_dataset(self) -> Dataset:
        """Raw GSM8K rows for GRPO.

        Important: GRPO needs access to the original GSM8K `answer` string that
        contains the '####' final-answer delimiter.
        """

        return self.load_raw_dataset_split("train")

    def load_dataset_split(self, split_name: str) -> Dataset:
        ds = self.load_raw_dataset_split(split_name)
        cols_to_remove = [c for c in ds.column_names if c != "text"]
        return ds.map(self.format_gsm8k, remove_columns=cols_to_remove)

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


    def convert_to_grpo(self, ds: Dataset) -> Dataset:
        ds = ds.map(self.grpo_processing, remove_columns=ds.column_names)

        # Drop rows with no extractable gold final answer.
        before_count = len(ds)
        ds = ds.filter(lambda x: x["answer"] is not None)
        dropped = before_count - len(ds)
        if dropped:
            print(f"[GSM8K] Warning: dropped {dropped} rows with answer=None (failed '####' extraction)")

        return ds

    @staticmethod
    def grpo_processing(x):
        """Convert GSM8K example to GRPO format: {prompt: str, question: str, answer: str}."""

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
        

        prompt = f"{system_prompt}\n\n{question}\n"

        return {
            "prompt": prompt,  # plain string prompt
            "question": question,  # for debugging/logging
            "answer": answer,  # gold final answer for reward functions
        }

