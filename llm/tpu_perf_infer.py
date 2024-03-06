import numpy as np
import torch
from tpu_perf.infer import SGInfer

np.random.seed(42)
torch.manual_seed(42)

def cos_sim(a, b):
  """计算两个向量a和b的余弦相似度"""

  a = np.array(a)
  b = np.array(b)

  inner_product = np.dot(a, b)
  # 内积
  norm_a = np.linalg.norm(a)
  norm_b = np.linalg.norm(b)
  # 模长
  cos_sim = inner_product / (norm_a * norm_b)

  return cos_sim

HIDDEN_SIZE = 4096
SEQ_LENGTH = 512
HEAD_DIM = 128
coeff = 100.0

input_states = coeff * torch.randn((1, 1, HIDDEN_SIZE)).numpy()
position_ids = 10*torch.tensor([range(1)], dtype=torch.int32).numpy().astype(np.int32)
attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH + 1), dtype=torch.float32).triu(diagonal=1).numpy()
history_k = coeff * torch.randn((SEQ_LENGTH, 1, 2, HEAD_DIM)).numpy()
history_v = coeff * torch.randn((SEQ_LENGTH, 1, 2, HEAD_DIM)).numpy()

x = dict()
x['input_states'] = input_states
x['position_ids'] = position_ids
x['attention_mask'] = attention_mask
x['history_k'] = history_k
x['history_v'] = history_v

model_0 = SGInfer("block_0.bmodel", devices=[16])

task_id = model_0.put(x)
task_id, int8_results, valid = model_0.get()


outputs = np.load("outputs.npy")
# print("FP32 Cosine Similarity:", cos_sim(outputs.flatten(), f32_results[0].flatten()))

# print("FP32&INT8 Cosine Similarity:", cos_sim(f32_results[0].flatten(), int8_results[0].flatten()))
