import torch
# example hyperparameters
n_embds = 32
dropout = 0.1
batch_size = 4
d_ff = 128
head_size = 2
N = 1
seq_len = 16

# # actual hyperarameters
# n_embds = 1024
# dropout = 0.1
# batch_size = 8
# d_ff = 2048
# head_size = 8
# N = 8
# seq_len = 16

# vocab used
vocab = ['0','1','2','3','4','5','6','7','8','9','+','[PAD]','[SOS]','[EOS]']


# vocab size
vocab_size = len(vocab)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
