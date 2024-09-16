import torch
import torch.nn as nn
import torch.nn.functional as F

def get_int(t):
    nums = [str(int(i.item())) for i in t[1:]]
    num = int(''.join(nums))
   
    return -num if t[0] == 13 else num

def split_int(num: int):
    return [13 if str(num)[0] == '-' else 14] + [int(i) for i in str(num)[1:]]

def add_zeros(num: list, sze: int):
    return [num[0]] + [0]*(sze - len(num)) + num[1:]


def create_addend(batch_size, digits):
    sign = torch.randint(13, 15, (batch_size, 1))
    addend = torch.randint(0, 9, (batch_size, digits))

    return torch.column_stack([sign, addend])

def preprocessed_batch(batch_size, digits):
    a = create_addend(batch_size, digits)
    b = create_addend(batch_size, digits)
    a = torch.column_stack([a, torch.full((batch_size,), 10)])
    sm = torch.cat([a, b], dim= 1)
    sm = torch.column_stack([sm, torch.full((batch_size,), 11)])
    sums = []
    for t in sm:
        first_half = get_int(t[:digits])
        second_half = get_int(t[digits+ 2:-1])
        sums.append(first_half + second_half)
        
    y = []
    for number in sums:
        splitednum = split_int(number)
        if len(splitednum) < (digits + 2):
            splitednum = add_zeros(splitednum, digits + 2)
        
        print(f"Splited: {splitednum}")
        y.append(splitednum)
        
    y = torch.tensor([i[::-1] for i in y])
    x = torch.column_stack([sm, y[:, :digits]])
    y = torch.column_stack([torch.full((batch_size,x.shape[1] - digits- 1), 12), y])
    return x, y

def get_val_data(samples: int, digits: int):
    
    x,y = preprocessed_batch(samples, digits)
    
    assert x.shape == y.shape, f"Shapes do not match: {x.shape} != {y.shape}"
    return x.long(),y.long()

def discriminate(t, specials):
    for i in t:
        for special in specials:
            if torch.all(torch.eq(i, special)):
                print(f"Special batch found: {i}")
                print("="*50)
                return False
    
    return True



def get_batch(batch_size, digits, specials):
    g = False
    
    # we ensure the model has never seen some numbers during training
    while not g:
        x, y = preprocessed_batch(batch_size, digits)
        # returns false if any tensor of x is in specials
        print(f"x {x}")
        print(f"y {y}")
        g = discriminate(x, specials)

    assert x.shape == y.shape, f"Shapes do not match: {x.shape} != {y.shape}"
    return x.long(), y.long()


print(get_batch(1, 5, torch.randint(0, 15, (1, (4 * 5 + 5)))))