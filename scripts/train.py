import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import GPT
import matplotlib.pyplot as plt
from data import get_batch, get_val_data
from dataclasses import dataclass

@dataclass
class Config:
    batch_size = 64
    # digits per number
    digits = 80
    block_size = (3 * digits) + 2
    n_embd = 384
    n_heads = 6
    n_blocks = 6
    vocab_size = 16  # 0-9 para dígitos, 10 para '+', 11 para '=', 12 para ignore_index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = 10

config = Config()
print(f"Using device: {config.device}")

model = GPT(config).to(config.device)

print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
opt = torch.optim.Adam(model.parameters(), lr=3e-4)


losses = []

specials, specials_labels = get_val_data(config.samples, config.digits)
import time
steps = 2000
for i in (range(steps)):
    t0 = time.time()
    x, y = get_batch(config.batch_size, config.digits, specials)
    
    x, y = x.to(config.device), y.to(config.device)

    logits, loss = model(x, y)
    losses.append(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000000 # in megaseconds
    if i % (steps//10) == 0:
        print(f"step {i}| loss: {loss.item()} | time {dt:.2f} ms")



plt.plot(losses)
plt.savefig("loss.png")
plt.show()

model.generate(specials, specials_labels, verbose=True)
