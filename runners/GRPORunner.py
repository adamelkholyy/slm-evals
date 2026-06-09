import os

from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer

from runners.PostTrainer import PostTrainer
from runners.rewards import (
    check_answer_correctness,
    check_numbers_extraction,
    match_format_exactly,
)
from settings import GRPO_CONFIG, system_prompt
from utils import save_model


class GRPORunner(PostTrainer):

    # custom dataset loading for GRPO
    def load_gsm8k(self):
        ds = load_dataset("openai/gsm8k", "main", split="train")
        return ds

    def run(self, model, tokenizer, args):
        ds = self.load_gsm8k()
        ds = self.convert_to_grpo(ds)

        config = dict(
            GRPO_CONFIG,
            output_dir=args.output_dir,
        )
        self.print_config(config)

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                match_format_exactly,
                check_answer_correctness,
                check_numbers_extraction,
            ],
            args=GRPOConfig(**config),
            train_dataset=ds,
        )
        trainer.train()
        save_model(trainer, "grpo")

    def convert_to_grpo(self, ds: Dataset) -> Dataset:
        ds = ds.map(
            self.grpo_processing,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )
        return ds

    @staticmethod
    def grpo_processing(x):
        """Convert GSM8K example to GRPO format."""

        question = x["question"]
        answer_text = x["answer"]
        answer = answer_text.split("####")[1].strip()

        # plain-text prompt (no chat template)
        prompt = f"{question}\n{system_prompt}"

        return {
            "prompt": prompt,
            "answer": answer,  # gold final answer for reward functions
        }
