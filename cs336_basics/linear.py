import numpy as np
import torch
import torch.nn as nn
# from fancy_einsum import einsum

# inherits from torch.nn.Module, performs a linear transformation
class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        """
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.W = nn.Parameter(
            torch.empty(
                (out_features, in_features), 
                device=device, 
                dtype=dtype
            )
        )
        mu, sigma = 0.0, np.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(
            self.W, mean=mu, std=sigma, a=-3*sigma, b=3*sigma
        )
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        apply linear transformation
        """
        # y = einsum("batch_size seq_len d_in, d_out d_in -> batch_size seq_len d_out", x, self.W)
        # return y
        y = x @ self.W.T
        return y


if __name__ == "__main__":
    d_in, d_out = 2, 2
    bs, sl = 1, 2
    
    lin = Linear(d_in, d_out)
    
    x = torch.randn((bs, sl, d_in))
    
    y = lin.forward(x)
    print(y)