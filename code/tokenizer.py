import re
special_tokens_pattern = r'\[(PAD|SOS|EOS)\]'

class MyTokenizer():
  def __init__(self):
    self.vocab = ['0','1','2','3','4','5','6','7','8','9','+','[PAD]','[SOS]','[EOS]']
    self.special_tokens = ['[',']','S','O','E','P','A','D']
    self.stoi = {s:i for i,s in enumerate(self.vocab)}
    self.itos = {i:s for i,s in enumerate(self.vocab)}
  def encode(self,s):
    non_special = [self.stoi[c] for c in s if c not in self.special_tokens]
    special = re.findall(special_tokens_pattern,s)
    special = ['['+ i + ']' for i in special]

    for i in special:
      if i == '[SOS]':
        non_special.insert(0,self.stoi['[SOS]'])
      elif i == '[EOS]':
        non_special.append(self.stoi['[EOS]'])
      elif i == '[PAD]':
        non_special.append(self.stoi['[PAD]'])
    return non_special


  def decode(self,i):
    non_special =  ''.join([self.itos[c] for c in i])
    return non_special

mytokenizer = MyTokenizer()