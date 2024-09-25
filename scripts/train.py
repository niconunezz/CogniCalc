import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import GPT
import matplotlib.pyplot as plt
from data import DataLoaderLite
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


train_loader = DataLoaderLite(config.batch_size, config.digits, config.samples)

torch.set_float32_matmul_precision('high')

model = GPT(config).to(config.device)

print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
opt = torch.optim.Adam(model.parameters(), lr=3e-4)



losses = []

import time
steps = 50
for i in (range(steps)):
    t0 = time.time()
    x, y = train_loader.get_batch()
    x, y = x.to(config.device), y.to(config.device)

    logits, loss = model(x, y)
    losses.append(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # in megaseconds
    # tokens_per_sec = (config.batch_size * config.block_size) / (dt/1000000)
    # print(f"step {i}| loss: {loss.item()} | time {dt:.2f} ms | tokens/s {tokens_per_sec:.2f}")
    if i % (steps//20) == 0:
        tokens_per_sec = (config.batch_size * config.block_size) / (dt/1000000)
        print(f"step {i}| loss: {loss.item()} | time {dt:.2f} ms | tokens/s {tokens_per_sec:.2f}")



plt.plot(losses)
plt.savefig("loss.png")
plt.show()

model.generate(train_loader.specials, train_loader.specials_labels, verbose=False)
