"""Encoder: chunks of tokens -> K memory vectors per chunk."""

import torch
from torch import nn

from engram.model.base import use_adapter
from engram.model.memory import MemoryProjection, MemoryQueries


class Encoder(nn.Module):
    def __init__(self, model, chunk_size: int, k: int, d: int, embed_rms: float):
        super().__init__()
        self.model = model  # shared PeftModel; not registered twice
        self.chunk_size, self.k = chunk_size, k
        self.queries = MemoryQueries(k, d, init_std=embed_rms)
        self.proj = MemoryProjection(d, embed_rms)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids [B, N*chunk_size] -> memory [B, N*K, d]."""
        b = input_ids.shape[0]
        chunks = input_ids.reshape(-1, self.chunk_size)  # [B*N, C]
        use_adapter(self.model, "encoder")
        tok = self.model.get_input_embeddings()(chunks)
        # Queries come after the chunk so causal attention lets them see all of it.
        x = torch.cat([tok, self.queries(chunks.shape[0])], dim=1)
        # Run the transformer body only (LoRA is injected in place); skips the unused LM head.
        body = self.model.get_base_model().model
        h = body(inputs_embeds=x).last_hidden_state[:, -self.k:]
        return self.proj(h).reshape(b, -1, h.shape[-1])
