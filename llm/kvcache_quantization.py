import numpy as np

class KVCacheQuantization():
    """A class to observe and record the max, min, and absolute max value of
    given tensor."""

    def __init__(self, num_head: int, head_dim: int) -> None:
        """Constructor for KVCacheObserver.

        Args:
            num_head : Number of heads
            head_dim : Dimension of each head
        """
        self.num_head = num_head
        self.head_dim = head_dim
        self.max_val = np.full((num_head, head_dim), -np.inf, dtype=np.float32)
        self.min_val = np.full((num_head, head_dim), np.inf, dtype=np.float32)
        self.absmax_val = np.full((num_head, head_dim), 0, dtype=np.float32)

    def observe(self, x):
        cur_max = x.squeeze(0).max(axis=0)
        cur_min = x.squeeze(0).min(axis=0)
        cur_absmax = np.abs(x.squeeze(0)).max(axis=0)

        self.max_val = np.maximum(self.max_val, cur_max)
        self.min_val = np.minimum(self.min_val, cur_min)
        self.absmax_val = np.maximum(self.absmax_val, cur_absmax)

    def quantize_cache(self, fdata):
        # b, s, head, h-dim->b, head, s, h-dim
        qtype = np.uint8
        shape = fdata.shape
        
        fdata_cal = fdata.squeeze(0)
        fmax = np.amax(fdata_cal, axis=0, keepdims=True)
        fmin = np.amin(fdata_cal, axis=0, keepdims=True)
        # Compute params
        scale = (fmax - fmin) / (self.max_val - self.min_val)
        zero = self.min_val - fmin / scale
        scale = np.expand_dims(scale, axis=0).repeat(shape[1], axis=1)
        zero = np.expand_dims(zero, axis=0).repeat(shape[1], axis=1)
        # Quantize
        res_data = fdata / scale + zero
        qdata = np.clip(res_data, self.min_val, self.max_val).astype(qtype)
        return qdata


key = np.load("key.npy")
value = np.load("value.npy")

k_observer = KVCacheQuantization(num_head=32, head_dim=128)
v_observer = KVCacheQuantization(num_head=32, head_dim=128)
k_observer.observe(key)
v_observer.observe(value)


q_key = k_observer.quantize_cache(key)
import pdb;pdb.set_trace()

