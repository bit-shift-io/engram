"""Batch-level mix of <full> reconstruction chunks and <terse> summary examples."""

import torch


class MixedSampler:
    def __init__(self, full_chunks: torch.Tensor, terse_examples: list[dict] | None,
                 full_ratio: float, batch_size: int, generator: torch.Generator):
        if full_ratio < 1.0 and not terse_examples:
            raise ValueError("full_ratio < 1 needs terse examples")
        self.full, self.terse = full_chunks, terse_examples or []
        self.full_ratio, self.batch_size, self.gen = full_ratio, batch_size, generator

    def next(self) -> tuple[str, object]:
        """('full', [B, C] tensor) or ('terse', list of B examples)."""
        if float(torch.rand((), generator=self.gen)) < self.full_ratio:
            idx = torch.randint(len(self.full), (self.batch_size,), generator=self.gen)
            return "full", self.full[idx]
        idx = torch.randint(len(self.terse), (self.batch_size,), generator=self.gen)
        return "terse", [self.terse[i] for i in idx.tolist()]
