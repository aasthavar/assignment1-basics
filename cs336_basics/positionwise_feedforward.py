import torch.nn as nn
import torch
from cs336_basics.linear import Linear

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.W1 = Linear(d_model, d_ff, **factory_kwargs)
        self.W2 = Linear(d_ff, d_model, **factory_kwargs)
        self.W3 = Linear(d_model, d_ff, **factory_kwargs)
    
    def _apply_silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.W2(self._apply_silu(self.W1(x)) * self.W3(x))
        return out
        