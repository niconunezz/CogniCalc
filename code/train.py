from model import Transformer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import *
from IPython.display import clear_output
from data_loader import get_batch
from tokenizer import mytokenizer as tokenizer

lusi = Transformer()
print('number of parameters: ',sum(p.numel() for p in lusi.parameters())/1e6,' M')
model = lusi.to(device)

optimizer = torch.optim.Adam(lusi.parameters(), lr=1e-7)
max_steps = 5000

for i in tqdm(range(max_steps)):
  src,tgt,label = get_batch(batch_size,seq_len)
  logits = model(src,tgt)
  B,T,C = logits.shape
  logits =logits.view(B*T,C)
  label = label.view(B*T)
  loss = F.cross_entropy(logits,label,ignore_index=tokenizer.encode('[PAD]')[0],label_smoothing=0.1)

  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  if i%500==0:
    numerical_loss = loss.item()
    print("\nloss: {:.3f}".format(numerical_loss))

  if i%2500==0:
    clear_output(wait=True)
    clear_output()