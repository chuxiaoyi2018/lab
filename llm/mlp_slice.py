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

def mlp(input_states, weight1, weight2, weight3):
    x = torch.nn.functional.layer_norm(input_states, [5120])
    x1 = torch.matmul(x, weight1)
    x1 = torch.nn.SiLU()(x1)
    x2 = torch.matmul(x, weight2)

    x1 = x1 * x2
    x1 = torch.matmul(x1, weight3)
    return x1 + input_states


def main():
    inputs = torch.randn(1,512,5120)
    weight1 = torch.randn(5120, 13824)
    weight2 = torch.randn(5120, 13824)
    weight3 = torch.randn(13824, 5120)

    print("-----------------------origin-----------------------")
    ori_output = mlp(inputs, weight1, weight2, weight3)
    print(ori_output[:,0,])

    print("-----------------------split-----------------------")
    split_output = mlp(inputs[:,0:1,:], weight1, weight2, weight3)
    print(split_output[:,:,])

main()