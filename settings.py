
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
    # Learning parameters optimized for reasoning tasks
    learning_rate=5e-6,  # Conservative LR to prevent destabilizing reasoning

    # Memory-efficient batch configuration
    per_device_train_batch_size=2,  # Small batch for GPU memory constraints
    gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 = 16

    # Sequence length limits for mathematical problems
    max_completion_length=512,  # room for step-by-step reasoning

    # Training duration and monitoring
    max_steps=500,  # increase further (2k-20k) for a full run
    logging_steps=1,  # log metrics every step for close monitoring

    # Stability and output configuration
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
