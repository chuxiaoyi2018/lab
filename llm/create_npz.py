import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(0)

HIDDEN_SIZE = 4096
SEQ_LENGTH = 512
HEAD_DIM = 128
coeff = 10.0


# hidden_states = torch.randn((1, 1, hidden_size))
# position_ids = torch.tensor([range(1)], dtype=torch.long)
# attention_mask = -1000 * torch.ones((1, 1, 1, MAX_LEN + 1), dtype=torch.float32).triu(diagonal=0)
# past_k = torch.randn((1, MAX_LEN, num_attention_heads, head_dim))
# past_v = torch.randn((1, MAX_LEN, num_attention_heads, head_dim))

input_states = coeff * torch.randn((1, 1, HIDDEN_SIZE)).numpy()
position_ids = torch.tensor([range(1)], dtype=torch.int32).numpy().astype(np.int32)
attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH + 1), dtype=torch.float32).triu(diagonal=1).numpy()
history_k = coeff * torch.randn((SEQ_LENGTH, 1, 2, HEAD_DIM)).numpy()
history_v = coeff * torch.randn((SEQ_LENGTH, 1, 2, HEAD_DIM)).numpy()

x = dict()
x['input_states'] = input_states
x['position_ids'] = position_ids
x['attention_mask'] = attention_mask
x['history_k'] = history_k
x['history_v'] = history_v

np.savez("inputs.npz", **x)
