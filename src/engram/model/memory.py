"""Trainable pieces around the frozen base: memory queries, projection, control embeddings."""

import torch
from torch import nn


class MemoryQueries(nn.Module):
    """K learned query embeddings appended after each chunk."""

    def __init__(self, k: int, d: int, init_std: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(k, d) * init_std)

    def forward(self, batch: int) -> torch.Tensor:
        return self.weight.unsqueeze(0).expand(batch, -1, -1)


class RMSNorm(nn.Module):
    def __init__(self, d: int, scale: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.full((d,), scale))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MemoryProjection(nn.Module):
    """[centre] + Linear (identity init) + RMSNorm scaled to the token-embedding RMS.

    Hidden states at the query slots are >90% a chunk-independent component (SmolLM2's massive
    activations), so after RMSNorm every chunk's memory looks the same. With k > 0 a per-slot
    running mean (EMA over training batches, fixed in eval) is subtracted first, leaving the
    chunk-specific part to fill the normalised vector.
    """

    def __init__(self, d: int, embed_rms: float, k: int = 0, momentum: float = 0.05):
        super().__init__()
        self.linear = nn.Linear(d, d)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.norm = RMSNorm(d, scale=embed_rms)
        self.momentum = momentum
        self.register_buffer("mean", torch.zeros(k, d) if k else None)
        self.register_buffer("seeded", torch.tensor(False))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h [B, K, d]."""
        if self.mean is not None:
            if self.training:
                with torch.no_grad():
                    batch_mean = h.detach().mean(0)
                    if not bool(self.seeded):
                        self.mean.copy_(batch_mean)
                        self.seeded.fill_(True)
                    else:
                        self.mean.lerp_(batch_mean, self.momentum)
            h = h - self.mean
        return self.norm(self.linear(h))


class ControlEmbedding(nn.Module):
    """Trainable rows for control tokens, kept apart from the frozen base embedding."""

    def __init__(self, token_ids: dict[str, int], init: torch.Tensor):
        super().__init__()
        self.index = {tid: i for i, tid in enumerate(token_ids.values())}
        self.weight = nn.Parameter(init.clone())

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        rows = torch.tensor([[self.index[int(t)] for t in row] for row in ids], device=ids.device)
        return self.weight[rows]
