import torch
import math
from cs336_basics.softmax import softmax


def sdpa(
    q: torch.Tensor, 
    k: torch.Tensor,
    v: torch.Tensor, 
    mask: torch.Tensor | None
) -> torch.Tensor:
    d_k = k.shape[-1]
    scores = (q @ k.mT) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf")) # mask == 0 -> masked; mask == 1 -> keep
    attention = softmax(scores, -1) @ v
    return attention