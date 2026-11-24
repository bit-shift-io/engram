"""Encoder: chunks of tokens -> K memory vectors per chunk."""

import torch
from torch import nn

from engram.model.base import use_adapter
from engram.model.memory import MemoryProjection, MemoryQueries


class Encoder(nn.Module):
    def __init__(self, model, chunk_size: int, k: int, d: int, embed_rms: float,
                 center: bool = False, segment_residual: bool = False, layer: int = -1,
                 mask_dims=(), pooled_residual: bool = False):
        super().__init__()
        self.model = model  # shared PeftModel; not registered twice
        self.chunk_size, self.k = chunk_size, k
        self.queries = MemoryQueries(k, d, init_std=embed_rms)
        self.proj = MemoryProjection(d, embed_rms, k=k if center else 0)
        self.embed_rms = embed_rms
        # Add the mean input embedding of each slot's C/K-token segment, RMS-scaled to the
        # embeddings: memory is chunk-specific from step 0, breaking the constant-prompt collapse.
        self.segment_residual = segment_residual
        # layer: hidden_states index to read (-1 = final, normed output); mask_dims: hidden dims
        # zeroed before projecting (SmolLM2's massive-activation dims, e.g. 17 and 507);
        # pooled_residual: add proj(mean of the chunk's own hidden states over each slot's segment).
        self.layer, self.mask_dims, self.pooled_residual = layer, tuple(mask_dims), pooled_residual
        if (segment_residual or pooled_residual) and chunk_size % k:
            raise ValueError("segment/pooled residual needs chunk_size divisible by k")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids [B, N*chunk_size] -> memory [B, N*K, d]."""
        b = input_ids.shape[0]
        chunks = input_ids.reshape(-1, self.chunk_size)  # [B*N, C]
        use_adapter(self.model, "encoder")
        tok = self.model.get_input_embeddings()(chunks)
        res = self.residual(tok) if self.segment_residual else None
        # Queries come after the chunk so causal attention lets them see all of it.
        x = torch.cat([tok, self.queries(chunks.shape[0])], dim=1)
        # Run the transformer body only (LoRA is injected in place); skips the unused LM head.
        body = self.model.get_base_model().model
        if self.layer == -1:
            hs = body(inputs_embeds=x).last_hidden_state
        else:
            hs = body(inputs_embeds=x, output_hidden_states=True).hidden_states[self.layer]
        if self.mask_dims:
            keep = torch.ones(hs.shape[-1], device=hs.device, dtype=hs.dtype)
            keep[list(self.mask_dims)] = 0
            hs = hs * keep
        h = hs[:, -self.k:]
        mem = self.proj(h)
        if self.pooled_residual:
            seg = hs[:, : self.chunk_size].reshape(hs.shape[0], self.k, -1, hs.shape[-1]).mean(2)
            mem = mem + self.proj(seg)
        if res is not None:
            mem = mem + res
        return mem.reshape(b, -1, h.shape[-1])

    def residual(self, tok: torch.Tensor) -> torch.Tensor:
        """tok [B, C, d] chunk input embeddings -> [B, K, d] RMS-scaled segment means."""
        seg = tok.reshape(tok.shape[0], self.k, -1, tok.shape[-1]).mean(2)
        return seg * torch.rsqrt(seg.pow(2).mean(-1, keepdim=True) + 1e-6) * self.embed_rms
