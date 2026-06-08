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
    match_format_exactly,
    check_answer_correctness,     # Mathematical accuracy
    check_numbers_extraction,     # Fallback number extraction
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


def get_model_and_tokenizer(model_name, use_lora=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # GRPO does left-padding for generation.
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    if use_lora:
        model = get_peft_model(model, LORA)
        model.print_trainable_parameters()
    else:
        print("LoRA disabled, training all parameters")
    return model, tokenizer


def save_model(trainer, label):
    is_peft = hasattr(trainer.model, "save_pretrained") and hasattr(trainer.model, "peft_config")
    out_dir = os.path.join(trainer.args.output_dir, f"{'adapter' if is_peft else 'checkpoint'}-{label}")
    trainer.model.save_pretrained(out_dir)
    trainer.processing_class.save_pretrained(out_dir)
    print(f"{'Adapter' if is_peft else 'Model'} saved to {out_dir}")


def run_sft(model, task, tok, cli_args):
    ds = task.load_dataset()
    common = dict(COMMON, output_dir=cli_args.output_dir)
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=SFTConfig(**common, dataset_text_field="text"),
    )
    trainer.train()
    save_model(trainer, "sft")


def run_grpo(model, task, tok, cli_args):
    # IMPORTANT: use raw GSM8K rows (question/answer) so we keep the '####' final-answer delimiter.
    ds = task.load_grpo_dataset()
    ds = task.convert_to_grpo(ds)


    common = dict(GRPO_CONFIG, output_dir=cli_args.output_dir)

    trainer = GRPOTrainer(
        model=model,  # LoRA-adapted model
        processing_class=tok,  # tokenizer (chat template applied automatically)
        reward_funcs=[
            match_format_exactly,
            check_answer_correctness,
            check_numbers_extraction,
        ],
        args=GRPOConfig(**common),
        train_dataset=ds,
    )
    trainer.train()
    save_model(trainer, "grpo")


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
    save_model(trainer, "dpo")


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
    save_model(trainer, "kto")


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
    save_model(trainer, "reward")


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
    default="Qwen/Qwen2.5-3B",
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

    lora = (args.method != "grpo")
    model, tok = get_model_and_tokenizer(args.model, use_lora=lora)
    task = Gsm8k()

    METHODS[args.method](model, task, tok, args)

    time_taken = time.time() - start
    hrs, rem = divmod(time_taken, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Completed in {int(hrs)}h {int(mins)}m {secs:.2f}s")
