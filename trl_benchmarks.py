import argparse
import re
import random
import time

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from trl import (
    SFTTrainer,
    SFTConfig,
    GRPOTrainer,
    GRPOConfig,
    DPOTrainer,
    DPOConfig,
    KTOTrainer,
    KTOConfig,
    RewardTrainer,
    RewardConfig,
)

MODEL = "EleutherAI/pythia-12b-deduped"

COMMON = dict(
    output_dir="./outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    bf16=True,
)


LORA = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)


def get_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
    model = get_peft_model(model, LORA)
    model.print_trainable_parameters()
    return model, tok


def preprocess_task(task):
    get_generic_negative = lambda x, prompt, answer: answer[::-1][:80]  # noqa: E731

    match task:
        case "gsm8k":
            ds = load_dataset("openai/gsm8k", "main", split="train")
            format_gsm8k = lambda x: {  # noqa: E731
                "text": f"Question: {x['question']}\nAnswer: {x['answer']}"
            }

            return ds.map(format_gsm8k), get_generic_negative

        case "arc":
            ds = load_dataset("ai2_arc", "ARC-Easy", split="train")

            # format MCQ choices for arc
            def format_arc(x):
                choices = "\n".join(
                    f"{l}. {t}"
                    for l, t in zip(x["choices"]["label"], x["choices"]["text"])
                )
                return {
                    "text": f"Question: {x['question']}\nChoices:\n{choices}\nAnswer: {x['answerKey']}"
                }

            def get_arc_negative(x, prompt, answer):
                choices = x["choices"]["text"]
                labels = x["choices"]["label"]
                correct_idx = labels.index(x["answerKey"])
                choices.pop(correct_idx)
                return random.choice(choices)

            return ds.map(format_arc), get_arc_negative

        case "hellaswag":
            ds = load_dataset("Rowan/hellaswag", split="train")

            # format MCQ ending options for hellaswag
            def format_hellaswag(x):
                endings = "\n".join(f"{i}. {e}" for i, e in enumerate(x["endings"]))
                return {
                    "text": f"Context: {x['ctx']}\nEndings:\n{endings}\nCorrect: {x['label']}"
                }

            def get_hellaswag_negative(x, prompt, answer):
                x["endings"].pop(int(x["label"]))
                return random.choice(x["endings"])

            return ds.map(format_hellaswag), get_hellaswag_negative

        case "piqa":
            ds = load_dataset("piqa", split="train")

            def format_piqa(x):
                return {
                    "text": (
                        f"Goal: {x['goal']}\n"
                        f"Solution 1: {x['sol1']}\n"
                        f"Solution 2: {x['sol2']}\n"
                        f"Correct: {x['label']}"
                    )
                }

            get_piqa_negative = lambda x, prompt, answer: not x["label"]  # noqa: E731
            return ds.map(format_piqa), get_piqa_negative

        case "code":
            ds = load_dataset("openai/openai_humaneval", split="test")  # 164 problems
            format_code = lambda x: {  # noqa: E731
                "text": f"# Task\n{x['prompt']}\n# Solution\n{x['canonical_solution']}"
            }
            return ds.map(format_code), get_generic_negative

        case _:
            raise ValueError(f"Unknown task: {task}")


# TODO: REDO THIS
_SPLIT_TOKENS = ["\nAnswer:", "\nCorrect:", "\nSolution\n", "\nEndings:\n"]


def _split_prompt_answer(text: str):
    """Return (prompt, answer) split at the natural task boundary."""
    for sep in _SPLIT_TOKENS:
        idx = text.rfind(sep)
        if idx != -1:
            return text[: idx + len(sep)], text[idx + len(sep) :]
    # fallback: split at last newline
    idx = text.rfind("\n")
    return (text[: idx + 1], text[idx + 1 :]) if idx != -1 else (text, "")


def to_preference(ds: Dataset, get_negative) -> Dataset:
    """(prompt, chosen, rejected) triples split at semantic boundaries."""

    def mk(x):
        prompt, answer = _split_prompt_answer(x["text"])
        rejected = get_negative(x, prompt, answer)
        return {"prompt": prompt, "chosen": answer, "rejected": rejected}

    return ds.map(mk, remove_columns=ds.column_names)


def to_kto(ds: Dataset, get_negative) -> Dataset:
    """KTO expects 'prompt', 'completion', 'label' (bool)."""
    rows = []
    for x in ds:
        prompt, answer = _split_prompt_answer(x["text"])
        rows.append({"prompt": prompt, "completion": answer, "label": True})
        rows.append(
            {
                "prompt": prompt,
                "completion": get_negative(x, prompt, answer),
                "label": False,
            }
        )
    return Dataset.from_list(rows)


def to_reward(ds: Dataset, tok, get_negative) -> Dataset:
    """RewardTrainer needs 'input_ids_chosen' / 'input_ids_rejected'."""

    def mk(x):
        prompt, answer = _split_prompt_answer(x["text"])
        chosen = tok(prompt + answer, truncation=True, max_length=512)
        rejected = tok(
            prompt + get_negative(x, prompt, answer), truncation=True, max_length=512
        )
        return {
            "input_ids_chosen": chosen["input_ids"],
            "attention_mask_chosen": chosen["attention_mask"],
            "input_ids_rejected": rejected["input_ids"],
            "attention_mask_rejected": rejected["attention_mask"],
        }

    return ds.map(mk, remove_columns=ds.column_names)


def gsm8k_reward(completions, prompts=None, **kw):
    """
    +1 if completion contains a number, 0 otherwise (proxy for math answer).
    """
    return [1.0 if re.search(r"\d+", c) else 0.0 for c in completions]


def run_sft(model, tok, ds, get_negative):
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=SFTConfig(**COMMON, dataset_text_field="text"),
    )
    trainer.train()


def run_grpo(model, tok, ds, get_negative):
    # GRPO needs a query column
    ds = ds.rename_column("text", "query")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[gsm8k_reward],
        train_dataset=ds,
        args=GRPOConfig(**COMMON),
    )
    trainer.train()


def run_dpo(model, tok, ds, get_negative):
    pref_ds = to_preference(tok, ds, get_negative)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=pref_ds,
        args=DPOConfig(**COMMON, precompute_ref_log_probs=True),
    )
    trainer.train()


def run_kto(model, tok, ds, get_negative):
    kto_ds = to_kto(ds, get_negative)
    trainer = KTOTrainer(
        model=model,
        train_dataset=kto_ds,
        args=KTOConfig(**COMMON),
    )
    trainer.train()


def run_reward(model, tok, ds, get_negative):
    rw_ds = to_reward(tok, ds, get_negative)
    trainer = RewardTrainer(
        model=model,
        train_dataset=rw_ds,
        args=RewardConfig(**COMMON),
    )
    trainer.train()


METHODS = {
    "sft": run_sft,
    "grpo": run_grpo,
    "dpo": run_dpo,
    "kto": run_kto,
    "reward": run_reward,
}


parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=list(METHODS), default="grpo")
parser.add_argument(
    "--task", choices=["gsm8k", "arc", "hellaswag", "piqa", "code"], default="gsm8k"
)
args = parser.parse_args()


if __name__ == "__main__":
    print(f"Benchmarking {MODEL}, method={args.method},  task={args.task}")
    start = time.time()

    model, tok = get_model_and_tokenizer()
    ds, get_negative = preprocess_task(args.task)
    METHODS[args.method](model, tok, ds, get_negative)

    time_taken = time.time() - start
    hrs, rem = divmod(time_taken, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Completed in {int(hrs)}h {int(mins)}m {secs:.2f}s")
