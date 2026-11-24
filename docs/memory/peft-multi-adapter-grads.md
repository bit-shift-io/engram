---
name: peft-multi-adapter-grads
description: peft set_adapter silently freezes the inactive LoRA — switch with use_adapter(); training needs grad clipping
metadata:
  type: convention
---

**Why:** The encoder and decoder share one base model with two named LoRA adapters (`encoder`, `decoder`). peft's `set_adapter(name)` sets `requires_grad=False` on the other adapter, so one adapter silently stops training while the loss still falls.
**How to apply:** Any code that switches adapters within one forward/backward pass, or builds param groups or optimizers over LoRA params.

- Switch with `engram.model.base.use_adapter(peft_model, name)`, which re-enables grads on all `lora_` params after `set_adapter`. Never call `set_adapter` directly.
- Keep a test asserting gradients reach both adapters.
- Clip gradients (max norm 1.0). Unclipped, lr 5e-3 diverges on SmolLM2-135M. Clipped, lr 2e-3 converges.
