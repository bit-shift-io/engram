"""Decoder: [memory][control][target] -> teacher-forced loss, or greedy generation."""

import contextlib

import torch
from torch import nn

from engram.model.base import use_adapter


class MemoryKVAdapter(nn.Module):
    """Low-rank deltas on every layer's k_proj/v_proj output at memory positions only.

    Target-token keys/values and all Q/O/MLP stay the frozen base, so these params change how
    memory is *read* but cannot store the target text. Forward hooks add the delta to the first
    `n_mem` sequence positions; n_mem is 0 (a no-op) except inside Decoder calls, so the encoder
    and KV-cached generation steps after the prefix are untouched.
    """

    def __init__(self, model, rank: int, alpha: float | None = None):
        super().__init__()
        self.n_mem = 0
        self.scale = (alpha or 2 * rank) / rank
        mods = [m for n, m in model.named_modules()
                if n.endswith("self_attn.k_proj") or n.endswith("self_attn.v_proj")]
        self.A, self.B = nn.ParameterList(), nn.ParameterList()
        for i, m in enumerate(mods):
            lin = m.base_layer if hasattr(m, "base_layer") else m
            d_in, d_out = lin.in_features, lin.out_features
            self.A.append(nn.Parameter(torch.randn(rank, d_in) / d_in ** 0.5))
            self.B.append(nn.Parameter(torch.zeros(d_out, rank)))
            m.register_forward_hook(self._hook(i))

    def _hook(self, i: int):
        def hook(module, args, out):
            n = self.n_mem
            if n <= 0:
                return None
            x = args[0][:, :n]
            delta = (x @ self.A[i].t()) @ self.B[i].t() * self.scale
            return torch.cat([out[:, :n] + delta, out[:, n:]], dim=1)
        return hook

    @contextlib.contextmanager
    def memory(self, n: int):
        prev, self.n_mem = self.n_mem, n
        try:
            yield
        finally:
            self.n_mem = prev


class Decoder(nn.Module):
    def __init__(self, model, control, use_lora: bool = True, input_dropout: float = 0.0,
                 dropout_id: int = 0, memory_adapter_rank: int = 0):
        super().__init__()
        self.mem_adapter = MemoryKVAdapter(model, memory_adapter_rank) if memory_adapter_rank else None
        # Training only: replace this fraction of teacher-forced target *inputs* (labels kept) by
        # dropout_id, so the decoder can't reconstruct from context alone and must read memory.
        self.input_dropout, self.dropout_id = input_dropout, dropout_id
        self.model = model
        self.control = control
        self.use_lora = use_lora  # False: run the frozen base (all adapters disabled)

    def _ctx(self):
        if self.use_lora:
            use_adapter(self.model, "decoder")
            return contextlib.nullcontext()
        return self.model.disable_adapter()

    def _mem(self, n: int):
        return self.mem_adapter.memory(n) if self.mem_adapter is not None else contextlib.nullcontext()

    def _prefix(self, memory: torch.Tensor, control_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([memory, self.control(control_ids)], dim=1)

    def _run(self, x: torch.Tensor, labels=None, attention_mask=None):
        # Default position ids start at 0, at the first memory vector.
        with self._ctx():
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
        inp = target_ids
        if self.training and self.input_dropout > 0:
            drop = torch.rand(target_ids.shape, device=target_ids.device) < self.input_dropout
            inp = target_ids.masked_fill(drop, self.dropout_id)
        tgt = self.model.get_input_embeddings()(inp)
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
        with self._mem(memory.shape[1]):
            return self._run(torch.cat([prefix, tgt], dim=1), labels, attn).loss

    @torch.no_grad()
    def generate(self, memory, control_ids, max_new_tokens: int, eos_id: int) -> list[int]:
        """Greedy decode for a single example (batch 1)."""
        x = self._prefix(memory, control_ids)
        emb = self.model.get_input_embeddings()
        out: list[int] = []
        past = None
        for _ in range(max_new_tokens):
            # Memory is only in the first (prefix) call; cached steps add new tokens only.
            with self._ctx(), self._mem(memory.shape[1] if past is None else 0):
                r = self.model(inputs_embeds=x, past_key_values=past, use_cache=True)
            past = r.past_key_values
            nxt = int(r.logits[0, -1].argmax())
            if nxt == eos_id:
                break
            out.append(nxt)
            x = emb(torch.tensor([[nxt]], device=x.device))
        return out
