---
name: prefix-control-token-ignored
description: A single prefix control token (<full>/<terse>) gets overridden by the input's genre on long outputs; use per-layer control vectors (control_vectors) and keep the data mix balanced
metadata:
  type: convention
---

**Why:** With only a `<full>` token before the target, on news articles the decoder wrote a one-line summary and stopped (~35 tokens instead of ~320). It had learned "article means summarise" from the data mix, and one token position couldn't overrule that.
**How to apply:** When adding a new control flag (density, task, slot), or when one output mode starts copying another.

- `control_vectors: true` adds a learned vector per flag to every decoder layer's residual stream at control + target positions (never memory, never the encoder). It's zero-initialised, so it's a no-op until trained.
- Don't overwrite fixed hidden dims as an "enum": the frozen base uses every dim, and SmolLM2 has huge outlier dims (17, 507).
- Keep the mix balanced from the start of a stage: if one input type only ever appears with one flag, the model keys on the input, not the flag. Starting from a checkpoint that already learned the shortcut didn't undo it.
- Every output mode must be trained on the memory sizes it will be evaluated on. `<full>` trained only on 1-chunk memory stopped early on 4-chunk memory (`multi_full_ratio`).
