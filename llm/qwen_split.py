import torch
B, nh, seq, dim = 4, 2, 1024, 64
splits = [521, 803, 1000, 1024] # 可以任意切分

q = torch.randn(B, nh, seq, dim)
k = torch.randn(B, nh, seq, dim)
v = torch.randn(B, nh, seq, dim)

def gen_causal_mask(seq_q_len, seq_kv_len):
    row_idx = torch.arange(seq_q_len).unsqueeze(-1)
    col_idx = torch.arange(seq_kv_len)
    r = col_idx < row_idx + seq_kv_len - seq_q_len + 1
    r = torch.logical_not(r)
    return r

def attention(q, k, v, causal_mask=False):
    # q, k, v: (B, nh, seq, dim)
    # attn: (B, nh, seq, seq)
    # out: (B, nh, seq, dim)
    attn = torch.einsum('bhid,bhjd->bhij', q, k) / dim**0.5
    if causal_mask:
        seq_q_len, seq_kv_len = q.shape[2], k.shape[2]
        mask = gen_causal_mask(seq_q_len, seq_kv_len)
        attn = attn.masked_fill(mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    out = torch.einsum('bhij,bhjd->bhid', attn, v)
    return out

out1 = attention(q, k, v, causal_mask=True)

## split seq_q
total_out = []
for i in range(len(splits)):
    split = splits[i]
    seq_q = q[:, :, 0:split, :] if i == 0 else q[:, :, splits[i-1]:split, :]
    seq_k = k[:, :, :split, :]
    seq_v = v[:, :, :split, :]
    out = attention(seq_q, seq_k, seq_v, causal_mask=True)
    total_out.append(out)
breakpoint()
out2 = torch.cat(total_out, dim=2)

print(torch.allclose(out1, out2, atol=1e-6))