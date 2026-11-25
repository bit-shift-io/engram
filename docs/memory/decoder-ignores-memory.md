---
name: decoder-ignores-memory
description: A trainable decoder LoRA learns to ignore soft-prompt memory; train the decoder frozen (plus a memory-only k/v adapter), or at a low LR once memory is in use
metadata:
  type: convention
---

**Why:** With a full-LR decoder LoRA, training loss falls just as well when memory is ignored. The decoder memorises the training text or learns a constant soft prompt, and real vs shuffled memory score the same.
**How to apply:** Any time you train the decoder side: a new stage, a new objective, or a config that turns `decoder_lora` on.

- Train reconstruction with `decoder_lora: false`. Read memory through `decoder_memory_adapter_rank` (rank 8 worked; rank 32 did not). That adapter changes keys/values at memory positions only, so it can't store the target text.
- When a later stage needs the decoder LoRA (e.g. summarising), start from a checkpoint where memory is already used, at a low LR (3e-4), with the encoder side lower still (~1e-4).
- More distinct training chunks helped more than more epochs: 2000 chunks × 6 epochs gave a 1.32-nat gap, against 0.28 for 512 × 17.
- Check with [[memory-necessity-eval]] at every checkpoint, not just at the end.
