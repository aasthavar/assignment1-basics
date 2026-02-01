import torch.nn as nn
import torch


class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        device = device or "cpu"
        
        pos = torch.arange(max_seq_len, device=device)
        k = torch.arange(0, d_k, 2, device=device) # this corresponds to 2k-2; k = [1,2,3,..., d_k/2], 2k-2 = [0, 2, 4, ... d_k]
        
        inv_freqs = 1.0 / (torch.pow(theta, k/d_k))
        angles = pos[:, None] * inv_freqs[None, :] # shape: [max_seq_len, d_k/2]
        
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x.shape = (..., d_k) (... = any batch dims + seq_len)
        # token_positions.shape = (...,) (same leading shape as x without d_k)
        # self.sin, self.cos shape = (max_seq_len, d_k//2)
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]
        
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos
        
        out = torch.empty_like(x)
        out[..., 0::2] = y_even
        out[..., 1::2] = y_odd
        
        return out
        
        