import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import GPT
import matplotlib.pyplot as plt
from data import get_batch, get_val_data


class Config:
    batch_size = 64
    # digits per number
    digits = 20
    # block_size = (4 * digits + 5)
    block_size = 3 *digits + 6
    n_embd = 384
    n_heads = 6
    n_blocks = 6
    vocab_size = 15  # 0-9 para dígitos, 10 para '+', 11 para '=', 12 para ignore_index, 13 para '-', 14 para '+'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = 10

config = Config()
print(f"Using device: {config.device}")

model = GPT(config).to(config.device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)


losses = []

specials, specials_labels = get_val_data(config.samples, config.digits)
for i in tqdm(range(2450)):
    x, y = get_batch(config.batch_size, config.digits, specials)
    
    x, y = x.to(config.device), y.to(config.device)

    logits, loss = model(x, y)
    losses.append(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()

    if i % 100 == 0:
        print(f"\nLoss: {loss.item()}")



plt.plot(losses)
plt.savefig("loss.png")
plt.show()

model.generate(get_batch, specials, specials_labels)
