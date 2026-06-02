from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-12b-v0",
    cache_dir="./pythia-12b-v0",
)

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-12b-v0",
    cache_dir="./pythia-12b-v0",
)


def call_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = model.generate(**inputs, max_length=20)
    return tokenizer.decode(tokens[0])


output = call_model("Hearts, Diamonds, Clubs, ")
print(output)
