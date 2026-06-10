import argparse
import time
import wandb

from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import LORA_CONFIG
from utils import resolve_output_dir

from runners.GRPORunner import GRPORunner
from runners.KTORunner import KTORunner
from runners.SFTRunner import SFTRunner


parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["sft", "grpo", "dpo", "kto", "reward"], default="grpo")
parser.add_argument(
    "--model",
    default="Qwen/Qwen2.5-3B",
)
parser.add_argument(
    "--run_name",
    default="run",
    help="Run name used to form output dir (default: <method>_<task>).",
)
parser.add_argument(
    "--output_dir",
    default=None,
    help="Explicit output directory (overrides --output_root/--run_name).",
)
args = parser.parse_args()


if __name__ == "__main__":
    run = wandb.init(
        entity="adamelkholy25-university-of-cambridge",
        project="dissertation",
        name=args.run_name
    )

    args.output_dir = resolve_output_dir(args)

    match args.method:
        case "sft": post_trainer = SFTRunner()
        case "grpo": post_trainer = GRPORunner()
        case "kto": post_trainer = KTORunner()

    print(
        f"Benchmarking {args.model}, method={args.method}, task=Gsm8k\n"
        f"Checkpoints/logs -> {args.output_dir}"
    )
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    if args.method == "grpo":
        print("Running GRPO: LoRA disabled, training all parameters")
    else:
        model = get_peft_model(model, LORA_CONFIG)
        model.print_trainable_parameters()

    post_trainer.run(model, tokenizer, args)

    time_taken = time.time() - start
    hrs, rem = divmod(time_taken, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Completed in {int(hrs)}h {int(mins)}m {secs:.2f}s")
