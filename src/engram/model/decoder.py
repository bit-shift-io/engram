"""Decoder: [memory][control][target] -> teacher-forced loss, or greedy generation."""

import contextlib
import re

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


class ControlVectors(nn.Module):
    """A learned vector per control flag (<full>, <terse>, ...) per decoder layer, added to that
    layer's output (the residual stream) at every position from `start` on: control and target
    positions, never memory. A prefix control token is one position the decoder can stop
    attending to; this flag is present at every step and layer. Zero-init, so a no-op until
    trained, and inactive (flags None) outside Decoder calls, so the encoder is untouched.
    """

    def __init__(self, model, control_token_ids: list[int]):
        super().__init__()
        self.layers = [m for n, m in model.named_modules() if re.search(r"(^|\.)layers\.\d+$", n)]
        self.index = {t: i for i, t in enumerate(control_token_ids)}
        d = model.config.hidden_size
        self.vec = nn.Parameter(torch.zeros(len(control_token_ids), len(self.layers), d))
        self.flags, self.start = None, 0
        for i, m in enumerate(self.layers):
            m.register_forward_hook(self._hook(i))

    def flag_index(self, control_ids: torch.Tensor) -> torch.Tensor:
        """[B] control token ids -> [B] flag indices."""
        return torch.tensor([self.index[int(t)] for t in control_ids], device=control_ids.device)

    def _hook(self, i: int):
        def hook(module, args, out):
            if self.flags is None:
                return None
            h = out[0] if isinstance(out, tuple) else out
            v = self.vec[self.flags, i][:, None]
            s = self.start
            h = torch.cat([h[:, :s], h[:, s:] + v], dim=1)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    @contextlib.contextmanager
    def active(self, flags, start: int):
        prev = self.flags, self.start
        self.flags, self.start = flags, start
        try:
            yield
        finally:
            self.flags, self.start = prev


def fact_weights(token_losses: torch.Tensor, mask: torch.Tensor, cap: float) -> torch.Tensor:
    """Per-token weights [B, T] from reference losses: detached, mean 1 per row over mask=1
    tokens, clipped at `cap`, renormalised to mean 1; 0 where mask is 0. Clip+renormalise is
    repeated so the result stays <= cap (renormalising lifts clipped tokens above it)."""
    m = mask.float()
    w = token_losses.detach().float() * m
    w = torch.where(w.sum(1, keepdim=True) > 0, w, m)  # all-zero row -> uniform
    n = m.sum(1, keepdim=True).clamp(min=1)
    w = w / (w.sum(1, keepdim=True) / n).clamp(min=1e-8)
    for _ in range(w.shape[1]):  # each pass clips >= 1 more token per row, or stops
        if w.max() <= cap + 1e-6:
            break
        clipped = w >= cap
        free = (w * ~clipped).sum(1, keepdim=True)
        budget = n - cap * clipped.sum(1, keepdim=True)  # mass left for unclipped tokens
        w = torch.where(clipped, cap * m, w * budget / free.clamp(min=1e-8))
    return w


class Decoder(nn.Module):
    def __init__(self, model, control, use_lora: bool = True, input_dropout: float = 0.0,
                 dropout_id: int = 0, memory_adapter_rank: int = 0,
                 control_token_ids: list[int] | None = None):
        super().__init__()
        self.mem_adapter = MemoryKVAdapter(model, memory_adapter_rank) if memory_adapter_rank else None
        self.ctrl_vec = ControlVectors(model, control_token_ids) if control_token_ids else None
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

    def _flags(self, flags, start: int):
        """Activate control vectors for flag indices [B], from sequence position `start` on."""
        if self.ctrl_vec is None:
            return contextlib.nullcontext()
        return self.ctrl_vec.active(flags, start)

    def _flag_ids(self, control_ids):
        return None if self.ctrl_vec is None else self.ctrl_vec.flag_index(control_ids[:, 0])

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
             target_mask=None, label_mask=None, weights=None) -> torch.Tensor:
        """memory_mask [B, M] / target_mask [B, T] (1 = real) allow right-padded batches.
        label_mask [B, T] (0 = attended but not supervised, e.g. an <answer> question prompt).
        weights [B, T] (optional): per-target-token loss weights; mean over supervised tokens."""
        if weights is not None:
            tl = self.token_losses(memory, control_ids, target_ids, memory_mask, target_mask,
                                   label_mask)
            sup = self._labels(target_ids, target_mask, label_mask) != -100
            return (tl * weights).sum() / sup.sum().clamp(min=1)
        return self._forward(memory, control_ids, target_ids, memory_mask, target_mask,
                             label_mask, use_labels=True).loss

    def token_losses(self, memory, control_ids, target_ids, memory_mask=None,
                     target_mask=None, label_mask=None) -> torch.Tensor:
        """Per-target-token cross-entropy [B, T]; 0 where the token is not supervised."""
        logits = self._forward(memory, control_ids, target_ids, memory_mask, target_mask,
                               label_mask, use_labels=False).logits
        n_pre, t = memory.shape[1] + control_ids.shape[1], target_ids.shape[1]
        pred = logits[:, n_pre - 1 : n_pre - 1 + t].float()
        labels = self._labels(target_ids, target_mask, label_mask)
        ce = nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]), labels.reshape(-1),
                                         ignore_index=-100, reduction="none")
        return ce.reshape(labels.shape)

    @staticmethod
    def _labels(target_ids, target_mask, label_mask):
        labels = target_ids if target_mask is None else target_ids.masked_fill(target_mask == 0, -100)
        if label_mask is not None:
            labels = labels.masked_fill(label_mask == 0, -100)
        return labels

    def _forward(self, memory, control_ids, target_ids, memory_mask, target_mask, label_mask,
                 use_labels: bool):
        prefix = self._prefix(memory, control_ids)
        inp = target_ids
        if self.training and self.input_dropout > 0:
            drop = torch.rand(target_ids.shape, device=target_ids.device) < self.input_dropout
            inp = target_ids.masked_fill(drop, self.dropout_id)
        tgt = self.model.get_input_embeddings()(inp)
        ignore = torch.full(prefix.shape[:2], -100, dtype=torch.long, device=target_ids.device)
        labels = torch.cat([ignore, self._labels(target_ids, target_mask, label_mask)], dim=1)
        attn = None
        if memory_mask is not None or target_mask is not None:
            b, dev = target_ids.shape[0], target_ids.device
            mm = memory_mask if memory_mask is not None else torch.ones(memory.shape[:2], device=dev)
            tm = target_mask if target_mask is not None else torch.ones_like(target_ids)
            ctrl = torch.ones(b, control_ids.shape[1], dtype=torch.long, device=dev)
            attn = torch.cat([mm.long(), ctrl, tm.long()], dim=1)
        with self._mem(memory.shape[1]), self._flags(self._flag_ids(control_ids), memory.shape[1]):
            return self._run(torch.cat([prefix, tgt], dim=1), labels if use_labels else None, attn)

    @torch.no_grad()
    def generate(self, memory, control_ids, max_new_tokens: int, eos_id: int,
                 question_ids: torch.Tensor | None = None) -> list[int]:
        """Greedy decode for a single example (batch 1). question_ids [1, Q] (the <answer> prompt)
        follows the control token; like in `loss`, it is covered by the control vector."""
        x = self._prefix(memory, control_ids)
        emb = self.model.get_input_embeddings()
        if question_ids is not None:
            x = torch.cat([x, emb(question_ids.to(x.device))], dim=1)
        out: list[int] = []
        past = None
        flags = self._flag_ids(control_ids)
        for _ in range(max_new_tokens):
            # Memory is only in the first (prefix) call; cached steps add new tokens only.
            start = memory.shape[1] if past is None else 0
            with self._ctx(), self._mem(start), self._flags(flags, start):
                r = self.model(inputs_embeds=x, past_key_values=past, use_cache=True)
            past = r.past_key_values
            nxt = int(r.logits[0, -1].argmax())
            if nxt == eos_id:
                break
            out.append(nxt)
            x = emb(torch.tensor([[nxt]], device=x.device))
        return out
