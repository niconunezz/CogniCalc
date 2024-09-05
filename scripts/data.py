import torch
import torch.nn as nn
import torch.nn.functional as F

def get_int(t):
    nums = [str(int(i.item())) for i in t]
    num = int(''.join(nums))
    return num

def split_int(num):
    return [int(i) for i in str(num)]

def add_zeros(num: list, sze):
    while len(num) < sze:
        num = [0] + num
    return num

def get_batch(batch_size, digits):

    a = torch.randint(0, 9, (batch_size, digits//2))
    b = torch.randint(0, 9, (batch_size, digits//2))
    a = torch.column_stack([a, torch.zeros((batch_size,)).fill_(10)])
    sm = torch.cat([a, b], dim=1)
    sm = torch.column_stack([sm, torch.zeros((batch_size,)).fill_(11)])
    sums = []
    for t in sm:
        sums.append(get_int(t[:digits//2]) + get_int(t[digits//2 + 1:-1]))
    
    c = []
    for number in sums:
        splitednum = split_int(number)
        if len(splitednum) < digits//2 + 1:
            splitednum = add_zeros(splitednum, digits//2 + 1)
        c.append(splitednum)
    
    c = torch.tensor([i[::-1] for i in c])
    sm = torch.column_stack([sm, c[:, :digits//2]])
    c = torch.column_stack([torch.zeros((batch_size, sm.shape[1] - (digits//2)- 1)).fill_(12), c])
    assert sm.shape == c.shape, f"Shapes do not match: {sm.shape} != {c.shape}"
    return sm.long(),c.long()

# x,y = get_batch(1, 10)
# print(x.shape)