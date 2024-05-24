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

def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor, begin, end, stride) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = hn // 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq, begin:end:stride]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)

def attention(query, key, value, o_weight, rope_cache, history_k, history_v, attention_mask, num_devices):
    present = []

    # value
    value = value.reshape(1,1,2,128)
    present.append(value)
    value = torch.cat((history_v, value), dim=0)
    value = value.permute(2,1,0,3)
    value = value.reshape(2,513,128)
    v1, v2 = value[:1], value[1:]

    # key
    key = key.reshape(1,1,2,128)
    key = apply_rotary_pos_emb(key, rope_cache, 0, 32, 1)
    present.append(key)
    key = torch.cat((history_k, key), dim=0)
    key = key.permute(2,1,3,0).reshape(2,128,513)
    k1, k2 = key[:1], key[1:]

    # query
    query = query.reshape(1,1,32 // num_devices,128)
    query = apply_rotary_pos_emb(query, rope_cache, 0, 32, 1)
    query = query.reshape(32 // num_devices, 1, 128)
    q1, q2 = query[:16 // num_devices], query[16 // num_devices:]

    # qk
    qxk1, qxk2 = torch.matmul(q1, k1), torch.matmul(q2, k2)
    # print(q1[0:1,0:1,:])
    qk = torch.cat((qxk1, qxk2), dim=0)
    qk = qk * 0.0883
    qk = qk.reshape(1, 32 // num_devices, 1, 513)
    qk = qk + attention_mask
    qk = torch.nn.functional.softmax(qk, dim=3)
    qk = qk.reshape(32 // num_devices, 1, 513)
    qk1, qk2 = qk[:16 // num_devices], qk[:16 // num_devices]

    # qkv
    qkv1, qkv2 = torch.matmul(qk1, v1), torch.matmul(qk2, v2)
    qkv = torch.cat((qkv1, qkv2), dim=0)
    qkv = qkv.reshape(1, 1, 32 // num_devices * 128)
    qkv = torch.matmul(qkv, o_weight)
    return qkv, present[0], present[1]

def inference_float():
    inputs = torch.randn(1,1,4096) * 0.001
    position_ids = torch.randint(0, 10, (1,1))
    attention_mask = -1000 * torch.ones((1, 1, 1, 512 + 1), dtype=torch.float32).triu(diagonal=0)
    rope_cache = torch.randn(512,32,2)
    history_k = torch.randn(512,1,2,128)
    history_v = torch.randn(512,1,2,128)

    q_weight = torch.randn(4096, 4096)
    k_weight = torch.randn(4096, 256)
    v_weight = torch.randn(4096, 256)
    o_weight = torch.randn(4096, 4096)

    print("-----------------------origin-----------------------")
    query = torch.matmul(inputs, q_weight.transpose(0,1))
    key = torch.matmul(inputs, k_weight)
    value = torch.matmul(inputs, v_weight)
    qkv, present_v, present_k = attention(query, key, value, o_weight.transpose(0,1), rope_cache, 
                                          history_k, history_v, attention_mask, num_devices=1)
    # print(qkv[:,:,:10])
    final = qkv + inputs
    print(final[:,:,:10])


    num_devices = 2
    final = 0
    print(f"-----------------------begin!!!-----------------------")
    for i in range(num_devices):
        print(f"-----------------------split {i}-----------------------")
        # q_weight_0 = q_weight[i * (4096 // num_devices): (i+1) * 4096 // num_devices]
        q_weight_0 = torch.cat([q_weight[i*2048//num_devices:(i+1)*2048//num_devices], q_weight[i*2048//num_devices+2048:(i+1)*2048//num_devices+2048]], dim=0)
        k_weight_0 = k_weight
        v_weight_0 = v_weight
        # o_weight_0 = o_weight[:, i * (4096 // num_devices): (i+1) * 4096 // num_devices]
        o_weight_0 = torch.cat([o_weight[:, i*2048//num_devices:(i+1)*2048//num_devices], o_weight[:, i*2048//num_devices+2048:(i+1)*2048//num_devices+2048]], dim=1)

        query_0 = torch.matmul(inputs, q_weight_0.transpose(0,1))
        key_0 = torch.matmul(inputs, k_weight_0)
        value_0 = torch.matmul(inputs, v_weight_0)
        history_k_0 = history_k
        history_v_0 = history_v

        qkv_0, present_v_0, present_k_0 = attention(query_0, key_0, value_0, o_weight_0.transpose(0,1), rope_cache, 
                                                    history_k_0, history_v_0, attention_mask, num_devices)
        # print(qkv_0[:,:,:10])
        final = final + qkv_0 + 0.5 * inputs
        print(final[:,:,:10])
    print(final[:,:,:10])


def weight_reorder(i, size, num_devices, dim, weight):
    return torch.cat([weight[i*size//num_devices:(i+1)*size//num_devices], weight[i*size//num_devices+size:(i+1)*size//num_devices+size]], dim=dim)

def weight_reorder2(i, size, num_devices, dim, weight):
    return torch.cat([weight[:, i*size//num_devices:(i+1)*size//num_devices], weight[:, i*size//num_devices+size:(i+1)*size//num_devices+size]], dim=dim)

def inference_int4():
    inputs = torch.randn(1,1,4096) * 0.001
    position_ids = torch.randint(0, 10, (1,1))
    attention_mask = -1000 * torch.ones((1, 1, 1, 512 + 1), dtype=torch.float32).triu(diagonal=0)
    rope_cache = torch.randn(512,32,2)
    history_k = torch.randn(512,1,2,128)
    history_v = torch.randn(512,1,2,128)

    q_weight = torch.randn(4096, 4096)
    k_weight = torch.randn(4096, 256)
    v_weight = torch.randn(4096, 256)
    o_weight = torch.randn(4096, 4096)
    q_zp, q_scale = torch.randn(4096, 4096), torch.randn(4096, 4096)
    o_zp, o_scale = torch.randn(4096, 4096), torch.randn(4096, 4096)
    


    print("-----------------------origin-----------------------")
    query = torch.matmul(inputs, (q_weight * q_scale - q_zp).transpose(0,1))
    key = torch.matmul(inputs, k_weight)
    value = torch.matmul(inputs, v_weight)
    qkv, present_v, present_k = attention(query, key, value, ((o_weight * o_scale) - o_zp).transpose(0,1), rope_cache, 
                                          history_k, history_v, attention_mask, num_devices=1)
    # print(qkv[:,:,:10])
    final = qkv + inputs
    print(final[:,:,:10])


    num_devices = 2
    final = 0
    print(f"-----------------------begin!!!-----------------------")
    for i in range(num_devices):
        print(f"-----------------------split {i}-----------------------")
        # q_weight_0 = q_weight[:, i * (4096 // num_devices): (i+1) * 4096 // num_devices]
        q_weight_0 = torch.cat([q_weight[i*2048//num_devices:(i+1)*2048//num_devices], q_weight[i*2048//num_devices+2048:(i+1)*2048//num_devices+2048]], dim=0)
        q_zp_0 = weight_reorder(i, 2048, num_devices, 0, q_zp)
        q_scale_0 = weight_reorder(i, 2048, num_devices, 0, q_scale)
        q_weight_0 = (q_weight_0 * q_scale_0) - q_zp_0

        k_weight_0 = k_weight
        v_weight_0 = v_weight


        o_weight_0 = torch.cat([o_weight[:, i*2048//num_devices:(i+1)*2048//num_devices], o_weight[:, i*2048//num_devices+2048:(i+1)*2048//num_devices+2048]], dim=1)
        o_zp_0 = weight_reorder2(i, 2048, num_devices, 1, o_zp)
        o_scale_0 = weight_reorder2(i, 2048, num_devices, 1, o_scale)
        o_weight_0 = (o_weight_0 * o_scale_0) - o_zp_0
        breakpoint()

        query_0 = torch.matmul(inputs, q_weight_0.transpose(0,1))
        key_0 = torch.matmul(inputs, k_weight_0)
        value_0 = torch.matmul(inputs, v_weight_0)
        history_k_0 = history_k
        history_v_0 = history_v

        qkv_0, present_v_0, present_k_0 = attention(query_0, key_0, value_0, o_weight_0.transpose(0,1), rope_cache, 
                                                    history_k_0, history_v_0, attention_mask, num_devices)
        # print(qkv_0[:,:,:10])
        final = final + qkv_0 + 0.5 * inputs
        print(final[:,:,:10])
    print(final[:,:,:10])
    pdb.set_trace()

def inference_kvcache_float():
    inputs = torch.randn(1,1,4096) * 0.001
    position_ids = torch.randint(0, 10, (1,1))
    attention_mask = -1000 * torch.ones((1, 1, 1, 512 + 1), dtype=torch.float32).triu(diagonal=0)
    rope_cache = torch.randn(512,32,2)
    history_k = torch.randn(512,1,2,128)
    history_v = torch.randn(512,1,2,128)

    q_weight = torch.randn(4096, 4096)
    k_weight = torch.randn(4096, 256)
    v_weight = torch.randn(4096, 256)
    o_weight = torch.randn(4096, 4096)

    print("-----------------------origin-----------------------")
    query = torch.matmul(inputs, q_weight.transpose(0,1))
    key = torch.matmul(inputs, k_weight)
    value = torch.matmul(inputs, v_weight)
    qkv, present_v, present_k = attention(query, key, value, o_weight.transpose(0,1), rope_cache, 
                                          history_k, history_v, attention_mask, num_devices=1)
    # print(qkv[:,:,:10])
    final = qkv + inputs
    print(final[:,:,:10])


    num_devices = 2
    final = 0
    print(f"-----------------------begin!!!-----------------------")
    for i in range(num_devices):
        print(f"-----------------------split {i}-----------------------")
        # q_weight_0 = q_weight[i * (4096 // num_devices): (i+1) * 4096 // num_devices]
        q_weight_0 = torch.cat([q_weight[i*2048//num_devices:(i+1)*2048//num_devices], q_weight[i*2048//num_devices+2048:(i+1)*2048//num_devices+2048]], dim=0)
        k_weight_0 = k_weight
        v_weight_0 = v_weight
        # o_weight_0 = o_weight[:, i * (4096 // num_devices): (i+1) * 4096 // num_devices]
        o_weight_0 = torch.cat([o_weight[:, i*2048//num_devices:(i+1)*2048//num_devices], o_weight[:, i*2048//num_devices+2048:(i+1)*2048//num_devices+2048]], dim=1)

        query_0 = torch.matmul(inputs, q_weight_0.transpose(0,1))
        key_0 = torch.matmul(inputs, k_weight_0)
        value_0 = torch.matmul(inputs, v_weight_0)
        history_k_0 = history_k
        history_v_0 = history_v

        qkv_0, present_v_0, present_k_0 = attention(query_0, key_0, value_0, o_weight_0.transpose(0,1), rope_cache, 
                                                    history_k_0, history_v_0, attention_mask, num_devices)
        # print(qkv_0[:,:,:10])
        final = final + qkv_0 + 0.5 * inputs
        print(final[:,:,:10])
    print(final[:,:,:10])