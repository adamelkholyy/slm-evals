import argparse
import time
import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model
from Gsm8k import Gsm8k

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


COMMON = dict(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    num_train_epochs=1,
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


def resolve_output_dir(cli_args):
    if cli_args.output_dir:
        out = cli_args.output_dir
    else:
        run_name = cli_args.run_name or f"{cli_args.method}_{cli_args.task}"
        out = os.path.join(cli_args.output_root, run_name)

    if os.path.exists(out):
        out = f"{out}-{int(time.time())}"

    os.makedirs(out, exist_ok=True)
    return out


def get_model_and_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = get_peft_model(model, LORA)
    model.print_trainable_parameters()
    return model, tok


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
    ds = task.load_dataset()
    ds = ds.rename_column("text", "prompt")  # GRPO formatting
    common = dict(COMMON, output_dir=cli_args.output_dir)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[task.reward_function],
        train_dataset=ds,
        args=GRPOConfig(**common),
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
