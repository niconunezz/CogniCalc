import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Hyperparameters

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape , weight=self.weight, bias=self.bias, eps=1e-5)

class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qkv = nn.Linear(config.n_embd, config.n_embd * 3)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.n_heads = config.n_heads

    def forward(self, x):
        B, T, C = x.shape
        # chunk is the same as split, but by definition just "attempts" to split the tensor.
        qkv = self.qkv(x).chunk(3, dim=-1)
        # divide in heads and transpose to get the right shape, making each head work as a batch that multiplies parallelly
        q, k, v = map(lambda t: t.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2), qkv) # (B, n_heads, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / C ** 0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v

        # recombine heads and project out
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )

    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, True)
        self.ln2 = LayerNorm(config.n_embd, True)
        self.attn = AttentionHead(config)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x



class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.device = config.device
        self.encoders = nn.Sequential(*[DecoderBlock(config) for _ in range(config.n_blocks)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.digits = config.digits
    def forward(self, x, targets=None):
        B, T = x.shape
        
        tok_emb = self.tok_emb(x) # B, T, C
        pos_emb = self.pos_emb(torch.arange(T).to(self.device)) # T, C
        
        x = tok_emb + pos_emb
        x = self.encoders(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), ignore_index= 12)
        
        return logits, loss

    def tensor_to_num(self, l):
        return int(''.join(map(str, l.tolist())))
    
    def tensor_in_list(self, tensor, tensor_list):
        return any(torch.equal(tensor, t) for t in tensor_list)

    def generate(self, get_batch, seen_batches):
        
        x, y = get_batch(1, self.digits)
        # we ensure the model has never seen the input during training
        while self.tensor_in_list(x, seen_batches):
            x, y = get_batch(1, self.digits)
            
        
        x = x[:, :(self.digits + 2)]
        first_half, second_half = map(lambda x: self.tensor_to_num(x), [x[0,:self.digits//2], x[0, self.digits//2+1:-1]])
        print(f"x: {first_half} + {second_half}")
        
        y = [i.item() for i in y[0, -(self.digits//2 + 1):]]
        print(f"should be {self.tensor_to_num(torch.tensor(y[::-1]))}")
        x = x.to(self.device)

        out = []
        for i in range((self.digits//2 + 1)):
            logits, _ = self.forward(x) # B, T, C
            logits = logits[:, -1, :] # B, C
            probs = F.softmax(logits, dim=-1) # B, C
            sample1 = torch.multinomial(probs, 1).squeeze().squeeze().item() # B
            out.append(int(sample1))
            x = torch.cat([x, torch.tensor([[sample1]]).to(self.device)], dim=-1)
            
        print(f'Accuracy: {((torch.tensor(out[::-1]) == torch.tensor(y[::-1])).count_nonzero().item() / len(out))*100}%')
        return self.tensor_to_num(torch.tensor(out[::-1]))


