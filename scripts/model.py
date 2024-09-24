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
            nn.ReLU()
        )
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT = 1


    def forward(self, x):
        return self.c_proj(self.net(x))

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
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.device = config.device
        self.decoders = nn.Sequential(*[DecoderBlock(config) for _ in range(config.n_blocks)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.tok_emb.weight = self.lm_head.weight # weight sharing

        self.apply(self._init_weights)
        
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANO_GPT_SCALE_INIT'):
                std *= (2 * self.config.n_blocks) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        
        tok_emb = self.tok_emb(x) # B, T, C
        pos_emb = self.pos_emb(torch.arange(T).to(self.device)) # T, C
        
        x = tok_emb + pos_emb
        x = self.decoders(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            
            # ignore_index= 12 is the padding token, key for masking the loss
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), ignore_index= 12)
        
        return logits, loss

    def tensor_to_num(self, l):
        return int(''.join(map(str, l.tolist())))
            

    def generate(self, specials, special_labels, verbose=True):
        acc_count = 0
        for x, y in zip(specials, special_labels):
            if verbose:
                print("="*50)
            first_half, second_half = map(lambda x: self.tensor_to_num(x), [x[:self.config.digits], x[self.config.digits+1:2*self.config.digits+1]])
            if verbose:
                print(f"x: {first_half} + {second_half}")
            
            y = [i.item() for i in y[-(self.config.digits+1):]]
            if verbose:
                print(f"should be {self.tensor_to_num(torch.tensor(y[::-1]))}")
            x = x.to(self.device)

            out = []
            
            logits, _ = self.forward(x.unsqueeze(0)) # B, T, C
            
            logits = logits[:, -(self.config.digits + 1):, :] # B, self.config.digits + 1, C
            probs = F.softmax(logits, dim=-1) # B, self.config.digits + 1, C
            
            sample = [torch.multinomial(probs[:,i], 1).squeeze().squeeze().item() for i in range(self.config.digits + 1)] # B, self.digits + 1
            
            for pred in sample:
                out.append(int(pred))
            
            out_tensor = torch.tensor(out[::-1])
            out =  self.tensor_to_num(out_tensor)
            
            acc = ((out_tensor == torch.tensor(y[::-1])).count_nonzero().item() / len(out_tensor))*100
            acc_count += acc
            if verbose:
                print(f'Accuracy: {acc}%')
                print(f"Model guessed: {out}")
                print("="*50)
            
        if not verbose:
            print(f"Averaged accuracy: {acc_count/len(specials)}%")
                


