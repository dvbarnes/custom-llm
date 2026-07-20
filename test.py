# uses the logic from train to load a model and run some experiments

import tiktoken
import torch

from gpt import GPTModel, generate_text_simple
from train import GPT_CONFIG_124M, text_to_token_ids, token_ids_to_text


model = GPTModel(GPT_CONFIG_124M)
state_dict = torch.load('output/model.pth')
model.load_state_dict(state_dict)

inference_device = torch.device("cpu")

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))