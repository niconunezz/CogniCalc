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

    a = torch.randint(0, 9, (batch_size, digits))
    b = torch.randint(0, 9, (batch_size, digits))
    a = torch.column_stack([a, torch.full((batch_size,), 10)])
    sm = torch.cat([a, b], dim= 1)
    sm = torch.column_stack([sm, torch.full((batch_size,), 11)])
    sums = []
    for t in sm:
        first_half = get_int(t[:digits])
        second_half = get_int(t[digits+ 1:-1])
        sums.append(first_half + second_half)
    
    y = []
    for number in sums:
        splitednum = split_int(number)
        if len(splitednum) < (digits + 1):
            splitednum = add_zeros(splitednum, digits + 1)
        y.append(splitednum)
    
    y = torch.tensor([i[::-1] for i in y])
    x = torch.column_stack([sm, y[:, :digits]])
    y = torch.column_stack([torch.full((batch_size,x.shape[1] - digits- 1), 12), y])

    assert x.shape == y.shape, f"Shapes do not match: {x.shape} != {y.shape}"
    return x.long(),y.long()

