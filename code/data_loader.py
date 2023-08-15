import random
import torch
from tokenizer import mytokenizer as tokenizer
from config import device

def get_raw_data(max_num):
  src,tgt = [],[]
  n = (max_num-1)//2
  max_numero = 10 ** (n) - 1
  n1 = random.randint(0,max_numero)
  n2 = random.randint(0,max_numero)

  for i in str(n1):
    src.append(i)
  src.append('+')

  for i in str(n2):
    src.append(i)

  lab = n1 + n2
  for i in str(lab):
    tgt.append(i)

  tgt = tgt[::-1]
  return src,tgt,max_num

# get_raw_data(10)
# output = (['6', '3', '2', '8', '+', '3', '4', '1', '1'], ['9', '3', '7', '9'], 10)

def get_mini_batch(seq_len):
  # obtain raw data
  seq_len = seq_len-2
  src,tgt,max_num = get_raw_data(seq_len)


  # tokenize
  tokeninized_src = tokenizer.encode(''.join(src))
  tokenized_tgt = tokenizer.encode(''.join(tgt))

  enc_diff = max_num-len(src)
  encoder_input = torch.cat(
      [torch.tensor(tokenizer.encode('[SOS]'),dtype=torch.int64),
      torch.tensor(tokeninized_src,dtype=torch.int64),
      torch.tensor(tokenizer.encode('[EOS]'),dtype=torch.int64),
      torch.tensor(tokenizer.encode('[PAD]')*enc_diff,dtype=torch.int64)]
  )
  dec_diff = max_num-len(tgt)+1
  decoder_input = torch.cat(
    [torch.tensor(tokenizer.encode('[SOS]'),dtype=torch.int64),
    torch.tensor(tokenized_tgt,dtype=torch.int64),
    torch.tensor(tokenizer.encode('[PAD]')*dec_diff,dtype=torch.int64)]
  )
  label_diff = max_num-len(tgt)+1
  label = torch.cat(
      [torch.tensor(tokenized_tgt,dtype=torch.int64),
      torch.tensor(tokenizer.encode('[EOS]'),dtype=torch.int64),
      torch.tensor(tokenizer.encode('[PAD]')*label_diff,dtype=torch.int64)]
  )

  return encoder_input,decoder_input,label


def get_batch(batch_size,seq_len):
  encoder = None
  for i in range(batch_size):
    encoder_input,decoder_input,label_input = get_mini_batch(seq_len)

    if encoder is not None:
      encoder = torch.cat([encoder,encoder_input])
      decoder = torch.cat([decoder,decoder_input])
      label = torch.cat([label,label_input])
    else:
      encoder = encoder_input
      decoder = decoder_input
      label = label_input




  encoder = (encoder.view(batch_size,-1)).to(device)

  decoder = (decoder.view(batch_size,-1)).to(device)
  label = (label.view(batch_size,-1)).to(device)

  return encoder,decoder,label