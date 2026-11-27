---
name: k32-fails-with-extra-objectives
description: Fresh K=32 encoders learn to use memory on plain WikiText reconstruction but stall when SQuAD mix or fact objectives are added
metadata:
  type: context
---

**Why:** Three K=32 runs from scratch stalled (as of 2026-11-27): the Stage 3b warm-up (SQuAD mix, val 2.94), and the fact-objective K=32 (4000 steps, val 3.01, real ROUGE-L 0.148 below shuffled 0.152, cloze real = shuffled). Only plain WikiText K=32 learned (val 2.56, real > shuffled). K=16 trained under the same objectives.
**How to apply:** Before starting any K=32 run with more than plain reconstruction.

- Check real vs shuffled memory early (first validation); equal losses mean the encoder is not learning.
- Cause unknown: learning rate, initialisation, or the memory adapter with 32 slots are candidates. Diagnose before spending another overnight run on K=32.
