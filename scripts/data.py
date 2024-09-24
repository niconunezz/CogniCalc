import torch
import torch.nn as nn
import torch.nn.functional as F

def get_int(tensor):
    nums = [int(''.join(map(str, row.tolist()))) for row in tensor]
    return nums

def split_int(num: int) -> list:
    return [int(i) for i in str(num)]

def add_zeros(num: list, sze: int) -> list:
    return [0]*(sze - len(num)) + num

def preprocessed_batch(batch_size: int, digits: int):
    
    ab = torch.randint(0, 9, (batch_size, digits*2))
    a, b = torch.split(ab, digits, dim= 1)

    a = torch.column_stack([a, torch.full((batch_size,), 10)])
    sm = torch.column_stack([a, b])
    sm = torch.column_stack([sm, torch.full((batch_size,), 11)])
    
    first_half = get_int(sm[: , :digits])
    second_half = get_int(sm[: , digits+ 1:-1])
    sums = [i + j for i, j in zip(first_half, second_half)]

    y = [add_zeros(split_int(number), digits + 1) for number in sums]
    
    y = torch.tensor([i[::-1] for i in y])
    x = torch.column_stack([sm, y[:, :digits]])
    y = torch.column_stack([torch.full((batch_size,x.shape[1] - digits- 1), 12), y]) # padding
    return x, y

def get_val_data(samples: int, digits: int):
    
    x,y = preprocessed_batch(samples, digits)

    assert x.shape == y.shape, f"Shapes do not match: {x.shape} != {y.shape}"
    return x.long(), y.long()

def discriminate(t, specials):
    for i in t:
        for special in specials:
            if torch.all(torch.eq(i, special)):
                print(f"Special batch found: {i}")
                print("="*50)
                return False
    
    return True


def get_batch(batch_size, digits, specials):

    x, y = preprocessed_batch(batch_size, digits)

    # if you want to prohibit the model to see val_data uncomment
    # spoiler: probability does it by itself

    # g = False
    # while not g:
    #     x, y = preprocessed_batch(batch_size, digits)
    #     # returns false if any tensor of x is in specials
    #     g = discriminate(x, specials)

    
    assert x.shape == y.shape, f"Shapes do not match: {x.shape} != {y.shape}"
    return x.long(),y.long()
