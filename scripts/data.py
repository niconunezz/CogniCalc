import torch
import torch.nn as nn
import torch.nn.functional as F

class DataLoaderLite:
    def __init__(self, batch_size, digits, samples, discriminate_var = False):
        self.batch_size = batch_size
        self.digits = digits
        self.samples = samples
        specials, specials_labels = self.get_val_data()
        self.specials = specials
        self.specials_labels = specials_labels
        self.discriminate_var = discriminate_var
    
    def get_int(self, tensor) -> list[int]:
        nums = [int(''.join(map(str, row.tolist()))) for row in tensor]
        return nums

    def split_int(self, num: int) -> list[int]:
        return [int(i) for i in str(num)]

    def add_zeros(self, num: list, sze: int) -> list[int]:
        return [0]*(sze - len(num)) + num
    
    def preprocessed_batch(self, val=False) -> tuple[torch.Tensor, torch.Tensor]:
        
        if val:
            B = self.samples
        else:
            B = self.batch_size

        ab = torch.randint(0, 9, (B, self.digits*2))
        a, b = torch.split(ab, self.digits, dim= 1)

        a = torch.column_stack([a, torch.full((B,), 10)])
        sm = torch.column_stack([a, b])
        sm = torch.column_stack([sm, torch.full((B,), 11)])
        
        first_half = self.get_int(sm[: , :self.digits])
        second_half = self.get_int(sm[: , self.digits+ 1:-1])
        sums = [i + j for i, j in zip(first_half, second_half)]

        y = [self.add_zeros(self.split_int(number), self.digits + 1) for number in sums]
        
        y = torch.tensor([i[::-1] for i in y])
        x = torch.column_stack([sm, y[:, :self.digits]])
        y = torch.column_stack([torch.full((B, x.shape[1] - self.digits- 1), 12), y]) # padding

        assert x.shape == y.shape, f"Shapes do not match: {x.shape} != {y.shape}"
        return x, y
    
    
    def get_val_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        x,y = self.preprocessed_batch(val=True)        
        return x.long(), y.long()

    def discriminate(self, t: torch.Tensor) -> bool:
        for i in t:
            for special in self.pecials:
                if torch.all(torch.eq(i, special)):
                    print(f"Special batch found: {i}")
                    print("="*50)
                    return False
        
        return True
    
    def get_batch(self):

        # if you want to prohibit the model to see val_data during training
        # spoiler: probability does it by itself

        if self.discriminate_var:
            g = False
            while not g:
                x, y = self.preprocessed_batch()
                # returns false if any tensor of x is in specials
                g = self.discriminate(x)
        else:
            x, y = self.preprocessed_batch()
        
        return x.long(),y.long()