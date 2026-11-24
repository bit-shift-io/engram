"""Decoder: [memory][control][target] -> teacher-forced loss, or greedy generation."""

import torch
from torch import nn

from engram.model.base import use_adapter


class Decoder(nn.Module):
    def __init__(self, model, control):
        super().__init__()
        self.model = model
        self.control = control

    def _prefix(self, memory: torch.Tensor, control_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([memory, self.control(control_ids)], dim=1)

    def _run(self, x: torch.Tensor, labels=None):
        # Default position ids start at 0, at the first memory vector.
        use_adapter(self.model, "decoder")
        return self.model(inputs_embeds=x, labels=labels)

    def loss(self, memory, control_ids, target_ids) -> torch.Tensor:
        prefix = self._prefix(memory, control_ids)
        tgt = self.model.get_input_embeddings()(target_ids)
        ignore = torch.full(prefix.shape[:2], -100, dtype=torch.long, device=target_ids.device)
        labels = torch.cat([ignore, target_ids], dim=1)
        return self._run(torch.cat([prefix, tgt], dim=1), labels).loss

    @torch.no_grad()
    def generate(self, memory, control_ids, max_new_tokens: int, eos_id: int) -> list[int]:
        """Greedy decode for a single example (batch 1)."""
        x = self._prefix(memory, control_ids)
        emb = self.model.get_input_embeddings()
        out: list[int] = []
        for _ in range(max_new_tokens):
            nxt = int(self._run(x).logits[0, -1].argmax())
            if nxt == eos_id:
                break
            out.append(nxt)
            x = torch.cat([x, emb(torch.tensor([[nxt]], device=x.device))], dim=1)
        return out
