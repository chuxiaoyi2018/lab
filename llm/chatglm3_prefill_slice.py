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
    max_seqlen = 1000
    seqlen_lists = [0, 400, 1000]

    top_left = torch.randn(2, 32, 1, max_seqlen) * 0.001
    top_right = torch.randn(max_seqlen, 2, 2, 128) * 0.001 # seq, batch, head, dim

    # origin
    right = top_right.transpose(0, 2) # head, batch, seq, dim
    # slice 0
    right_0 = right[:1]
    right_0 = right_0.reshape(2, 1, max_seqlen, 128)
    left_0 = top_left[:, :16]
    output_0 = torch.matmul(left_0, right_0)

    # slice 1
    right_1 = right[1:]
    right_1 = right_1.reshape(2, 1, max_seqlen, 128)
    left_1 = top_left[:, 16:]
    output_1 = torch.matmul(left_1, right_1)
    ori_output = torch.cat((output_0, output_1), dim=1)
    
    print(ori_output)

    output_slice = []
    for i in range(len(seqlen_lists)-1):
        seqlen = seqlen_lists[i+1] - seqlen_lists[i]
        top_left_slice = top_left[:, :, :, seqlen_lists[i]:seqlen_lists[i+1]]
        top_right_slice = top_right[seqlen_lists[i]:seqlen_lists[i+1]]

        right = top_right_slice.transpose(0, 2) # head, batch, seq, dim
        # slice 0
        right_0 = right[:1]
        right_0 = right_0.reshape(2, 1, seqlen, 128)
        left_0 = top_left_slice[:, :16]
        output_0 = torch.matmul(left_0, right_0)

        # slice 1
        right_1 = right[1:]
        right_1 = right_1.reshape(2, 1, seqlen, 128)
        left_1 = top_left_slice[:, 16:]
        output_1 = torch.matmul(left_1, right_1)
        output = torch.cat((output_0, output_1), dim=1)
        output_slice.append(output)
    breakpoint()


    # valid
    print((out - output).sum())
    breakpoint()

inference()