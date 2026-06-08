import argparse
import time
import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import get_peft_model
from Gsm8k import Gsm8k
from settings import COMMON, GRPO_CONFIG, LORA

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

from rewards import (
    match_format_exactly,         # Perfect structure compliance
    match_format_approximately,   # Partial format credit
    check_answer_correctness,     # Mathematical accuracy
    check_numbers_extraction,
)


def resolve_output_dir(cli_args):
    if cli_args.output_dir:
        out = cli_args.output_dir
    else:
        run_name = cli_args.run_name or f"{cli_args.method}_gsm8k"
        out = os.path.join("outputs", run_name)

    if os.path.exists(out):
        out = f"{out}-{int(time.time())}"

    os.makedirs(out, exist_ok=True)
    return out


def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # GRPO does left-padding for generation.
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = get_peft_model(model, LORA)
    model.print_trainable_parameters()
    return model, tokenizer


def save_adapter(trainer, label):
    adapter_dir = os.path.join(trainer.args.output_dir, f"adapter-{label}")
    trainer.model.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")


def run_sft(model, task, tok, cli_args):
    ds = task.load_dataset()
    common = dict(COMMON, output_dir=cli_args.output_dir)
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=SFTConfig(**common, dataset_text_field="text"),
    )
    trainer.train()
    save_adapter(trainer, "sft")


def run_grpo(model, task, tok, cli_args):
    # IMPORTANT: use raw GSM8K rows (question/answer) so we keep the '####' final-answer delimiter.
    ds = task.load_grpo_dataset()
    ds = task.convert_to_grpo(ds)

    # Sanity-check: if prompt is conversational (list of {role, content}), TRL will try to apply a chat template.
    # We want plain strings for GSM8K to avoid `tokenizer.chat_template` entirely.
    if len(ds) > 0:
        assert isinstance(ds[0]["prompt"], str), f"Expected ds['prompt'] to be str, got {type(ds[0]['prompt'])}"

    # Catch missing gold answers early.
    sample_n = min(20, len(ds))
    if sample_n:
        assert all(x["answer"] is not None for x in ds.select(range(sample_n))), (
            "answer=None found in first rows — check GSM8K '####' extraction in grpo_processing"
        )

    common = dict(GRPO_CONFIG, output_dir=cli_args.output_dir)

    trainer = GRPOTrainer(
        model=model,  # LoRA-adapted model
        processing_class=tok,  # IMPORTANT: plain tokenizer (no chat template needed)
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            check_answer_correctness,
            check_numbers_extraction,
        ],
        args=GRPOConfig(**common),
        train_dataset=ds,
    )
    trainer.train()
    save_adapter(trainer, "grpo")


def run_dpo(model, task, tok, cli_args):
    ds = task.load_dataset()
    ds = task.convert_to_preference(ds)
    common = dict(COMMON, output_dir=cli_args.output_dir)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=ds,
        args=DPOConfig(**common, precompute_ref_log_probs=True),
    )
    trainer.train()
    save_adapter(trainer, "dpo")


def run_kto(model, task, tok, cli_args):
    ds = task.load_dataset()
    ds = task.convert_to_kto(ds)
    common = dict(COMMON, output_dir=cli_args.output_dir)
    trainer = KTOTrainer(
        model=model,
        train_dataset=ds,
        args=KTOConfig(**common),
    )
    trainer.train()
    save_adapter(trainer, "kto")


def run_reward(model, task, tok, cli_args):
    ds = task.load_dataset()
    ds = task.convert_to_reward(ds, tok)
    common = dict(COMMON, output_dir=cli_args.output_dir)
    trainer = RewardTrainer(
        model=model,
        train_dataset=ds,
        args=RewardConfig(**common),
    )
    trainer.train()
    save_adapter(trainer, "reward")


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
    "--model",
    default="EleutherAI/pythia-12b-deduped",
)
parser.add_argument(
    "--run_name",
    default=None,
    help="Run name used to form output dir (default: <method>_<task>).",
)
parser.add_argument(
    "--output_dir",
    default=None,
    help="Explicit output directory (overrides --output_root/--run_name).",
)

args = parser.parse_args()


if __name__ == "__main__":
    args.output_dir = resolve_output_dir(args)
    print(
        f"Benchmarking {args.model}, method={args.method}, task=Gsm8k\n"
        f"Checkpoints/logs -> {args.output_dir}"
    )
    start = time.time()

    model, tok = get_model_and_tokenizer(args.model)
    task = Gsm8k()

    METHODS[args.method](model, task, tok, args)

    time_taken = time.time() - start
    hrs, rem = divmod(time_taken, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Completed in {int(hrs)}h {int(mins)}m {secs:.2f}s")
