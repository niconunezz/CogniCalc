import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# tokenizer
from tokenizer import mytokenizer as tokenizer
# hyperparameters
from config import *



class FeedForward(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embds,n_embds ),
      nn.ReLU(),
      nn.Linear(n_embds,n_embds ),
      nn.Dropout(dropout),
    )
  def forward(self, x):
    return self.net(x)

class Attention(nn.Module):
  def __init__(self):
    super().__init__()

    self.key = nn.Linear(n_embds, n_embds)
    self.query = nn.Linear(n_embds, n_embds)
    self.value = nn.Linear(n_embds, n_embds)
    self.proj = nn.Linear(n_embds, n_embds)
    self.dropout = nn.Dropout(dropout)
  def forward(self,q,k,v,mask):
    B,T,C = q.shape
    assert C % head_size == 0, "C % head_size != 0, make them divisible"
    q = self.query(q).view(B,T,head_size,C//head_size).transpose(1,2)
    k = self.key(k).view(B,T,head_size,C//head_size).transpose(1,2)
    v = self.value(v).view(B,T,head_size,C//head_size).transpose(1,2)

    mm = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))

    if mask is not None:
      mm = mm.masked_fill(mask==0,float('-inf'))


    attn = F.softmax(mm, dim=-1)


    y = (attn @ v).transpose(1,2).contiguous().view(B,T,C)

    y = self.proj(y)

    return self.dropout(y)

class EncoderBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.att= Attention()
    self.ff = FeedForward()
    self.norms = nn.ModuleList([nn.LayerNorm(n_embds) for _ in range(2)])
    self.dropout = nn.Dropout(dropout)
  def forward(self, x, mask):
    x = self.norms[0](x)
    x = x + self.att(x,x,x,mask)
    x = self.norms[1](x)
    x = x + self.ff(x)
    return self.dropout(x)

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.tok_emb = nn.Embedding(vocab_size,n_embds)
    self.pos_emb = nn.Embedding(seq_len,n_embds)
    self.blocks = nn.ModuleList([EncoderBlock() for _ in range(N)])
  def forward(self, x, mask):
    B,T = x.shape
    x = self.tok_emb(x) # B,T,C
    pos = self.pos_emb(torch.arange(T,device=device)) # B,T
    x = x + pos # B,T,C
    for block in self.blocks:
      x = block(x,mask)

    return x # B,T,C

class DecoderBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.sa = Attention()
    self.ca = Attention()
    self.ff = FeedForward()
    self.norms = nn.ModuleList([nn.LayerNorm(n_embds) for _ in range(3)])
    self.dropout = nn.Dropout(dropout)
  def forward(self, x, enc, mask):
    x = self.norms[0](x)
    x = x + self.sa(x,x,x,mask)
    x = self.norms[1](x)
    x = x + self.ca(x,enc,enc,mask=None)
    x = self.norms[2](x)
    x = x + self.ff(x)
    return self.dropout(x)

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.tok_emb = nn.Embedding(vocab_size,n_embds)
    self.pos_emb = nn.Embedding(seq_len,n_embds)
    self.blocks = nn.ModuleList([DecoderBlock() for _ in range(N)])

  def forward(self, x, enc, mask):
    B,T = x.shape
 
    x = self.tok_emb(x)
    pos = self.pos_emb(torch.arange(T,device=device))
    x = x + pos
    for block in self.blocks:
      x = block(x,enc,mask)
    return x

class FinalProj(nn.Module):
  def __init__(self):
    super().__init__()
    self.proj = nn.Linear(n_embds, vocab_size)
  def forward(self, x):
    return self.proj(x)


class Transformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.enc = Encoder()
    self.dec = Decoder()
    self.proj = FinalProj()
    self.padding_idx = tokenizer.encode('[PAD]')[0]
    self.eos_idx = tokenizer.encode('[EOS]')[0]
    self.sos_idx = tokenizer.encode('[SOS]')[0]
  def src_mask(self,src):

    mask = (src!=self.padding_idx).unsqueeze(1).unsqueeze(2)
    return mask.to(device)
  def tgt_mask(self,tgt):
    mask = torch.tril(torch.ones(seq_len,seq_len))
    return mask.to(device)
  
  def sum_mask(self,tgt):
    copy = tgt.clone()
    copy[:,-1] = 1e9
    mask =( tgt != 1e9).unsqueeze(1).unsqueeze(2)
    return mask.to(device)

  def encode(self,src):
    return self.enc(src,self.src_mask(src))

  def decode(self,decoder_input,encoder_output):
    return self.dec(decoder_input,encoder_output,self.sum_mask(decoder_input))

  def project(self,x):
    return self.proj(x)


  def forward(self,src,tgt):
    src_mask = self.src_mask(src)
    tgt_mask = self.tgt_mask(tgt)
    enc = self.enc(src,src_mask)
    dec = self.dec(tgt,enc,tgt_mask)
    return self.proj(dec)

  # function that takes in the tensor with the sum in it and return its predictions
  def suma(self,input):

    encoder_out = self.encode(input)
    
    decoder_input = torch.empty(batch_size,seq_len).fill_(self.sos_idx).type_as(input).to(device)
    dumb = False
    while dumb == False:
      decoder_input = decoder_input[:,-seq_len:]
      out = self.decode(decoder_input,encoder_out)
      
      probs = (self.project(out))[:,-1,:] # B,1,C
      
      next_token = torch.argmax(probs, dim=-1, keepdim=True) # B,1
      
      decoder_input = torch.cat([decoder_input, next_token],dim=1) #B,T+1
      
      for b in next_token:
        
        if b.item() == self.eos_idx:
          print(b.item())
          print(self.eos_idx)
          dumb = True
          
      

    return decoder_input.squeeze(0)
