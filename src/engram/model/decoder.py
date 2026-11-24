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

    def _run(self, x: torch.Tensor, labels=None, attention_mask=None):
        # Default position ids start at 0, at the first memory vector.
        use_adapter(self.model, "decoder")
        if attention_mask is None:
            return self.model(inputs_embeds=x, labels=labels)
        # Padded positions are skipped, so real positions stay contiguous from 0.
        pos = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        return self.model(inputs_embeds=x, labels=labels, attention_mask=attention_mask,
                          position_ids=pos)

    def loss(self, memory, control_ids, target_ids, memory_mask=None,
             target_mask=None) -> torch.Tensor:
        """memory_mask [B, M] / target_mask [B, T] (1 = real) allow right-padded batches."""
        prefix = self._prefix(memory, control_ids)
        tgt = self.model.get_input_embeddings()(target_ids)
        ignore = torch.full(prefix.shape[:2], -100, dtype=torch.long, device=target_ids.device)
        labels = target_ids if target_mask is None else target_ids.masked_fill(target_mask == 0, -100)
        labels = torch.cat([ignore, labels], dim=1)
        attn = None
        if memory_mask is not None or target_mask is not None:
            b, dev = target_ids.shape[0], target_ids.device
            mm = memory_mask if memory_mask is not None else torch.ones(memory.shape[:2], device=dev)
            tm = target_mask if target_mask is not None else torch.ones_like(target_ids)
            ctrl = torch.ones(b, control_ids.shape[1], dtype=torch.long, device=dev)
            attn = torch.cat([mm.long(), ctrl, tm.long()], dim=1)
        return self._run(torch.cat([prefix, tgt], dim=1), labels, attn).loss

    @torch.no_grad()
    def generate(self, memory, control_ids, max_new_tokens: int, eos_id: int) -> list[int]:
        """Greedy decode for a single example (batch 1)."""
        x = self._prefix(memory, control_ids)
        emb = self.model.get_input_embeddings()
        use_adapter(self.model, "decoder")
        out: list[int] = []
        past = None
        for _ in range(max_new_tokens):
            r = self.model(inputs_embeds=x, past_key_values=past, use_cache=True)
            past = r.past_key_values
            nxt = int(r.logits[0, -1].argmax())
            if nxt == eos_id:
                break
            out.append(nxt)
            x = emb(torch.tensor([[nxt]], device=x.device))
        return out
