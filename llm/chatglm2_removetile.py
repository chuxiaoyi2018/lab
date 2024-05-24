import torch
import random
import numpy as np
import pdb

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(42)

def inference():
    top_left = torch.randn(2, 32, 1, 513) * 0.001
    top_right = torch.randn(513, 2, 2, 128) * 0.001 # seq, batch, head, dim

    # origin
    left = top_left.reshape(64, 1, 513)
    right = top_right.unsqueeze(3)
    right = right.tile(1, 1, 1, 16, 1)
    right = right.reshape(513, 64, 128)
    right = right.transpose(0, 1)

    out = torch.matmul(left, right).reshape(2, 32, 1, 128)
    print(out)

    # remove tile
    right = top_right.transpose(0, 2) # head, batch, seq, dim
    # slice 0
    right_0 = right[:1]
    right_0 = right_0.reshape(2, 1, 513, 128)
    left_0 = top_left[:, :16]
    output_0 = torch.matmul(left_0, right_0)

    # slice 1
    right_1 = right[1:]
    right_1 = right_1.reshape(2, 1, 513, 128)
    left_1 = top_left[:, 16:]
    output_1 = torch.matmul(left_1, right_1)
    
    output = torch.cat((output_0, output_1), dim=1)
    print(output)

    # valid
    print((out - output).sum())
    breakpoint()

inference()