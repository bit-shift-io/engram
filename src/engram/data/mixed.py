"""Batch-level mix of <full> reconstruction chunks and <terse> summary examples."""

import torch


class MixedSampler:
    def __init__(self, full_chunks: torch.Tensor, terse_examples: list[dict] | None,
                 full_ratio: float, batch_size: int, generator: torch.Generator,
                 multi_full: list[dict] | None = None, multi_ratio: float = 0.0,
                 answer: list[dict] | None = None, answer_ratio: float = 0.0,
                 squad_full: list[dict] | None = None, squad_ratio: float = 0.0):
        """multi_ratio: fraction of <full> draws taken from multi_full (multi-chunk articles).
        squad_ratio: fraction of <full> draws taken from squad_full (whole SQuAD paragraphs),
        checked before multi_ratio, which then splits the remaining <full> draws.
        answer_ratio: fraction of all draws that are <answer> examples; full_ratio splits the rest."""
        if full_ratio < 1.0 and not terse_examples:
            raise ValueError("full_ratio < 1 needs terse examples")
        self.full, self.terse = full_chunks, terse_examples or []
        self.full_ratio, self.batch_size, self.gen = full_ratio, batch_size, generator
        self.multi, self.multi_ratio = multi_full or [], multi_ratio
        if multi_ratio > 0 and not self.multi:
            raise ValueError("multi_ratio > 0 needs multi_full examples")
        self.answer, self.answer_ratio = answer or [], answer_ratio
        if answer_ratio > 0 and not self.answer:
            raise ValueError("answer_ratio > 0 needs answer examples")
        self.squad, self.squad_ratio = squad_full or [], squad_ratio
        if squad_ratio > 0 and not self.squad:
            raise ValueError("squad_ratio > 0 needs squad_full examples")

    def next(self) -> tuple[str, object]:
        """('full', [B, C] tensor), ('multi', list of B <full> article examples),
        ('squad_full', list of B <full> SQuAD paragraph examples) or
        ('terse', list of B examples) or ('answer', list of B examples)."""
        if self.answer_ratio and float(torch.rand((), generator=self.gen)) < self.answer_ratio:
            idx = torch.randint(len(self.answer), (self.batch_size,), generator=self.gen)
            return "answer", [self.answer[i] for i in idx.tolist()]
        if float(torch.rand((), generator=self.gen)) < self.full_ratio:
            if self.squad_ratio and float(torch.rand((), generator=self.gen)) < self.squad_ratio:
                idx = torch.randint(len(self.squad), (self.batch_size,), generator=self.gen)
                return "squad_full", [self.squad[i] for i in idx.tolist()]
            if self.multi_ratio and float(torch.rand((), generator=self.gen)) < self.multi_ratio:
                idx = torch.randint(len(self.multi), (self.batch_size,), generator=self.gen)
                return "multi", [self.multi[i] for i in idx.tolist()]
            idx = torch.randint(len(self.full), (self.batch_size,), generator=self.gen)
            return "full", self.full[idx]
        idx = torch.randint(len(self.terse), (self.batch_size,), generator=self.gen)
        return "terse", [self.terse[i] for i in idx.tolist()]
