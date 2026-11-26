---
name: reconstruction-mix-dilutes-wikitext
description: Mixing SQuAD paragraph chunks into reconstruction training stalled a K=32 warm-up; keep the full WikiText set and strip EOS padding when adding a source
metadata:
  type: convention
---

**Why:** Replacing half of the WikiText reconstruction chunks with SQuAD paragraph chunks stopped a K=32 encoder from learning to use memory: val loss plateaued at ~2.94 and real vs shuffled memory scored almost the same. The same config on WikiText only escaped the plateau and reached 2.56.
**How to apply:** Any time a new source is mixed into reconstruction training (`mix_chunks`, `squad_recon_share`, or a new dataset).

- Add new-source chunks on top of the existing distinct WikiText chunks; never swap them in at a fixed total. Distinct chunk count matters more than epochs (see [[decoder-ignores-memory]]).
- SQuAD paragraphs end in a mostly-EOS last chunk: ~26% of SQuAD chunk tokens were padding, and a quarter of chunks were over half padding. Drop or trim those before training on them.
- K=32 from scratch sat on a val-loss plateau (~2.9) for ~1200 steps before improving; K=16 did not. Budget extra steps, and don't judge a K=32 run before then.
