import torch
import torch.nn as nn
from einops import rearrange
from cs336_basics.rope import RoPE
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import sdpa

class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None):
        super().__init__()
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        assert d_model % num_heads == 0
        
        if theta and max_seq_len:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
        else:
            self.rope = None
        
        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)

    def forward(self, x, token_positions=None):
        # x.shape = [batch_size, seq_len, d_model]
        Q = self.W_Q(x) # shape: [batch_szie, seq_len, d_model]
        K = self.W_K(x)
        V = self.W_V(x)
        mask = torch.tril(torch.ones(1, 1, Q.shape[-2], Q.shape[-2])).bool()
        
        q = rearrange(Q, "batch_size seq_len (h d_k) -> batch_size h seq_len d_k", h=self.num_heads)
        k = rearrange(K, "batch_size seq_len (h d_k) -> batch_size h seq_len d_k", h=self.num_heads)
        v = rearrange(V, "batch_size seq_len (h d_v) -> batch_size h seq_len d_v", h=self.num_heads)
        if self.rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
            
        o = sdpa(q, k, v, mask)
        O = rearrange(o, "batch_size h seq_len d_k -> batch_size seq_len (h d_k)")
        return self.W_O(O)