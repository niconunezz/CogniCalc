import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import GPT
import matplotlib.pyplot as plt
from data import get_batch


class Config:
    batch_size = 64
    # the nums being added will have (digits//2) digits
    digits = 80
    block_size = (3 * digits)//2 + 2
    n_embd = 384
    n_heads = 6
    n_blocks = 6
    vocab_size = 13  # 0-9 para dígitos, 10 para '+', 11 para '=', 12 para ignore_index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
print(f"Using device: {config.device}")

model = GPT(config).to(config.device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

for i in tqdm(range(1200)):
    x, y = get_batch(config.batch_size, config.digits)
    x, y = x.to(config.device), y.to(config.device)

    logits, loss = model(x, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if i % 100 == 0:
        print(f"\nLoss: {loss.item()}")
        
examples = 5
for i in range(examples):
    print("="*50)
    out = model.generate(get_batch)
    print(f"Model guessed: {out}")
    print("="*50)