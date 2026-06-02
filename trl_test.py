from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

model_name = "EleutherAI/pythia-12b-deduped"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)

dataset = load_dataset(
    "openai/gsm8k",
    "main",
    split="train"
)

def format_example(example):
    return {
        "text": (
            f"Question: {example['question']}\n"
            f"Answer: {example['answer']}"
        )
    }

dataset = dataset.map(format_example)

config = SFTConfig(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=20,
    logging_steps=10,
    save_steps=1000,
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    # tokenizer=tokenizer,
    train_dataset=dataset,
    args=config,
    # dataset_text_field="text",
)

trainer.train()


# for IT/chat-tuned models
'''
def format_chat(example):
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
 
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    return {"text": text}    
'''

# # LORA
# from peft import LoraConfig

# peft_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     target_modules="all-linear",
#     task_type="CAUSAL_LM",
# )

# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     args=config,
#     peft_config=peft_config,
#     dataset_text_field="text",
# )