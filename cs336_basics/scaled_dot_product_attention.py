import torch
import math
from cs336_basics.softmax import softmax


def sdpa(
    q: torch.Tensor, # [batch_size, ..., seq_len, d_k]
    k: torch.Tensor, # [batch_size, ..., seq_len, d_k]
    v: torch.Tensor, # [batch_size, ..., seq_len, d_v]
    mask: torch.Tensor | None  # [seq_len, seq_len]
) -> torch.Tensor:
    d_k = k.shape[-1]
    scores = (q @ k.mT) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf")) # mask == 0 -> -inf; mask == 1 -> keep
    attention = softmax(scores, -1) @ v
    return attention