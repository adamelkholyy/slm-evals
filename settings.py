from peft import LoraConfig

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

GRPO_CONFIG = dict(
    learning_rate=5e-6,  # Conservative LR to prevent destabilizing reasoning
    per_device_train_batch_size=2,  # Small batch for GPU memory constraints
    gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 = 16
    gradient_checkpointing=True,
    max_completion_length=1024,  # room for step-by-step reasoning
    max_steps=500,  # increase further (2k-20k) for a full run
    num_train_epochs=1,
    logging_steps=1,  # log metrics every step for close monitoring
    save_steps=500,
    bf16=True,  # mixed precision
    max_grad_norm=0.1,  # aggressive gradient clipping for stable training
)


LORA = LoraConfig(
    r=16,                              # Rank: adaptation capacity (16 good for reasoning tasks)
    lora_alpha=32,                     # Scaling factor (typically 2x rank)
    target_modules="all-linear", # Focus on attention query/value for reasoning
    lora_dropout=0.1,                  # Regularization to prevent overfitting
    bias="none",                       # Skip bias adaptation for simplicity
    task_type="CAUSAL_LM",             # Causal language modeling task
)

# Debug configuration
DEBUG_EVERY = 5           # print debug info every N steps
DEBUG_N = 1                # number of samples to print per debug step
DEBUG_PROMPT_CHARS = 800   # max chars to show for prompts
DEBUG_COMPLETION_CHARS = 1200  # max chars to show for completions
DEBUG_SHOW_FULL_PROMPT = False  # whether to show the full prompt untruncated