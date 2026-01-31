import torch.nn as nn
import torch
import einops

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_size,
        device=None,
        dtype=None
    ):
        """
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.W_embed = nn.Parameter(
            torch.empty(
                size=(num_embeddings, embedding_size),
                device=device,
                dtype=dtype
            )
        )
        mu, sigma = 0.0, 1.0
        nn.init.trunc_normal_(
            self.W_embed, mean=mu, std=sigma, a=-3*sigma, b=3*sigma
        )
    
    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        """embedding lookup"""
        return self.W_embed[token_ids]
    
    # def forward(
    #     self,
    #     token_ids: torch.Tensor
    # ) -> torch.Tensor:
    #     """embedding lookup"""
    #     # token_ids.shape = [batch_size, seq_len]
    #     out = self.W_embed[:token_ids.size(1), :]
    #     out = einops.repeat(out, "seq_len d_model -> batch_size seq_len d_model", batch_size=token_ids.size(0))
    #     return out  # out.shape = [batch_szie, seq_len, d_model]

if __name__ == "__main__":
    d_vocab, d_model = 512, 64
    batch_size, seq_len = 5, 20
    token_ids = torch.randint(0, d_vocab, (batch_size, seq_len))
    embed = Embedding(d_vocab, d_model)
    out = embed(token_ids)
    assert out.shape == (batch_size, seq_len, d_model)
    