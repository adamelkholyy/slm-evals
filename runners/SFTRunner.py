from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer

from runners.WandbCallback import WandbCallback
from runners.PostTrainer import PostTrainer
from settings import COMMON, system_prompt
from utils import save_model


class SFTRunner(PostTrainer):

    # custom dataset loading for SFT
    def load_gsm8k(self) -> Dataset:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        cols_to_remove = [
            c for c in ds.column_names if c not in ("prompt", "completion")
        ]
        return ds.map(
            self.format_gsm8k, remove_columns=cols_to_remove, load_from_cache_file=False
        )

    @staticmethod
    def format_gsm8k(x) -> dict:
        """Format as prompt/completion for completion-only SFT."""
        prompt = f"{system_prompt}\n\nQuestion: {x['question']}\nAnswer:"
        completion = f" {x['answer']}"  # leading space so it tokenises nicely
        return {"prompt": prompt, "completion": completion}

    def run(self, model, _tokenizer, args):
        ds = self.load_gsm8k()

        config = dict(
            COMMON,
            output_dir=args.output_dir,
        )
        self.print_config(config)

        trainer = SFTTrainer(
            model=model,
            train_dataset=ds,
            #callbacks=[WandbCallback()],
            args=SFTConfig(**config),
        )
        trainer.train()
        save_model(trainer, "sft")
