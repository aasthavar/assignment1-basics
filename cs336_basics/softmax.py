import torch

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    max_values = x.max(dim=i, keepdim=True).values
    norm_x = x - max_values
    exp_x = torch.exp(norm_x)
    sum_exp_x = exp_x.sum(dim=i, keepdim=True)
    softmax_x = exp_x / sum_exp_x
    return softmax_x

