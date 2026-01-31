import torch
import torch.nn as nn
import numpy as np

from fancy_einsum import einsum

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        """
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.empty((d_model), device=device, dtype=dtype))
        nn.init.normal_(self.g, mean=0.0, std=1.0)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: [batch_size, seq_len, d_model]
        in_dtype = x.dtype
        # upcast input to prevent overflow when squaring
        x = x.to(torch.float32)
        # calculate mean of squares (l2 norm) along the dimension of d_model
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # normalize
        x_norm = x / rms
        # rescale with g (gain) parameter
        out = x_norm * self.g
        # downcast 
        out = out.to(in_dtype)
        return out


if __name__ == "__main__":
    d_model = 512
    eps = 1e-5
    norm = RMSNorm(d_model, eps)
    
    bs, sl = 5, 20
    input = torch.rand((bs, sl, d_model))
    out = norm(input)
    assert out.shape == (bs, sl, d_model)
    # print(out)
    