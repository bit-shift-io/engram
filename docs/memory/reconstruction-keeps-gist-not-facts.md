---
name: reconstruction-keeps-gist-not-facts
description: Plain reconstruction training stores the gist of a chunk but drops names, numbers and dates; answer recall needs an objective that rewards facts
metadata:
  type: context
---

**Why:** Memory trained only on token-averaged reconstruction gets recon token F1 ~0.2–0.26 on SQuAD dev paragraphs but answer recall of only 5–7.5% (K=16 and K=32 alike, 2026-11-26). A downstream QA stage can't recover facts that memory never stored.
**How to apply:** Before relying on memory for anything fact-specific (QA, `<answer>`), or when choosing a training objective for the encoder.

- The loss averages over all tokens, and the frozen decoder predicts most of them from context alone. Fact tokens are a small, rare share, so missing them barely moves the loss, and the decoder fills the gaps with plausible guesses.
- Raising K from 16 to 32 did not help: both warm-ups reached 5% answer recall on 40 dev paragraphs.
- Measure fact retention with the `recall` eval suite (answer recall), not reconstruction loss or token F1.
- As of 2026-11-26 the planned fix is a fact-weighted objective: weight reconstruction tokens by how badly the decoder predicts them with zeroed memory, plus a masked-fact (cloze) task from memory during warm-up.
