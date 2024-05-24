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
NUM_ATTENTION_HEADS = 32
coeff = 1.0
SHARE_LENGTH = 6144
UNSHARE_LENGTH = 895
BATCH_SIZE = 2


# input_states = coeff * torch.zeros((BATCH_SIZE, 1, HIDDEN_SIZE)).numpy()
# position_ids = 10*torch.tensor(BATCH_SIZE*[range(1)], dtype=torch.int32).numpy().astype(np.int32)
# share_attention_mask = torch.zeros((1, 1, 1, SHARE_LENGTH), dtype=torch.float32).triu(diagonal=1).numpy()
# unshare_attention_mask = torch.zeros((BATCH_SIZE, 1, 1, UNSHARE_LENGTH), dtype=torch.float32).triu(diagonal=1).numpy()
# share_past_k = coeff * torch.randn((1, SHARE_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).numpy()
# share_past_v = coeff * torch.randn((1, SHARE_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).numpy()
# unshare_past_k = coeff * torch.randn((BATCH_SIZE, UNSHARE_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).numpy()
# unshare_past_v = coeff * torch.randn((BATCH_SIZE, UNSHARE_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).numpy()
# share_attention_mask[:,:,:,:10] = 1.
inputs = np.load("/workspace/LLM-TPU/models/Qwen/share_cache_demo/inputs.npz")
input_states = inputs["input_states"]
position_ids = inputs["position_ids"]
share_attention_mask = inputs["share_attention_mask"]
unshare_attention_mask = inputs["unshare_attention_mask"]
share_attention_mask = np.where(share_attention_mask<0, -10000., 0.).astype(np.float32)
unshare_attention_mask = np.where(unshare_attention_mask<0, -10000., 0.).astype(np.float32)
# share_attention_mask = torch.ones((1, 1, 1, SHARE_LENGTH), dtype=torch.float32).numpy()
# unshare_attention_mask = torch.zeros((BATCH_SIZE, 1, 1, UNSHARE_LENGTH), dtype=torch.float32).triu(diagonal=1).numpy()
share_past_k = inputs["share_past_k"]
share_past_v = inputs["share_past_v"]
unshare_past_k = inputs["unshare_past_k"]
unshare_past_v = inputs["unshare_past_v"]

model_0 = SGInfer("/workspace/LLM-TPU/models/Qwen/share_cache_demo/tmp/int4_1dev/block/block_share_cache_0.bmodel", devices=[2])

task_id = model_0.put(input_states, position_ids, share_attention_mask, unshare_attention_mask, share_past_k, share_past_v, unshare_past_k, unshare_past_v)
task_id, results_0, valid = model_0.get()


self_attention_mask = torch.zeros((1, 1, 1, 1)).numpy().astype(np.float32)
attention_mask = np.concatenate((share_attention_mask, unshare_attention_mask[:1], self_attention_mask), axis=-1)
past_k = np.concatenate((share_past_k, unshare_past_k[:1]), axis=1)
past_v = np.concatenate((share_past_v, unshare_past_v[:1]), axis=1)
model_1 = SGInfer("/workspace/LLM-TPU/models/Qwen/share_cache_demo/tmp/int4_1dev/block/block_cache_0.bmodel", devices=[2])
task_id = model_1.put(input_states[:1], position_ids[:1], attention_mask, past_k, past_v)
task_id, results_1, valid = model_1.get()

print(cos_sim(results_0[0][:1].flatten(), results_1[0].flatten()))
# print("FP32 Cosine Similarity:", cos_sim(outputs.flatten(), f32_results[0].flatten()))

# print("FP32&INT8 Cosine Similarity:", cos_sim(f32_results[0].flatten(), int8_results[0].flatten()))
