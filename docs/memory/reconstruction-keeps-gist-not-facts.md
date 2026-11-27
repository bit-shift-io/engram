---
name: reconstruction-keeps-gist-not-facts
description: Plain reconstruction training stores the gist of a chunk but drops names, numbers and dates; answer recall needs an objective that rewards facts (SQuAD cloze works)
metadata:
  type: context
---

**Why:** Memory trained only on token-averaged reconstruction gets recon token F1 ~0.2–0.26 on SQuAD dev paragraphs but answer recall of only 5–7.5% (K=16 and K=32 alike, 2026-11-26). A downstream QA stage can't recover facts that memory never stored.
**How to apply:** Before relying on memory for anything fact-specific (QA, `<answer>`), or when choosing a training objective for the encoder.

- The loss averages over all tokens, and the frozen decoder predicts most of them from context alone. Fact tokens are a small, rare share, so missing them barely moves the loss, and the decoder fills the gaps with plausible guesses.
- Raising K from 16 to 32 did not help: both warm-ups reached 5% answer recall on 40 dev paragraphs.
- Measure fact retention with the `recall` eval suite (answer recall), not reconstruction loss or token F1.
- Fact weighting plus a WikiText cloze task (2026-11-27, K=16, 3000 steps) did not fix it: answer recall stayed 7.5%, though recon F1 rose to 0.31.
- That run's cloze gap (real vs shuffled 1.3 nats) held on WikiText but not on SQuAD. On the `cloze` eval suite (40 SQuAD dev paragraphs, every gold answer blanked), that checkpoint scores answer loss real 3.48 / shuffled 4.60 / zeroed 4.21 nats. Older figures of 5.0 / 5.5 / 6.6 came from a lost script and don't compare.
- Training the cloze task on SQuAD train paragraphs, blanking gold answers (2026-11-27, K=16, 2000 more steps, 0.3 of batches), fixed it: answer recall rose to 0.20, and the cloze real-minus-zeroed gap went from 0.73 to 1.73 nats. Train the cloze task on the data you test on.
- Re-scored 2026-11-28 with exact-word answer matching (the old substring match let "5" match "50"): baseline (fact K=16) answer recall 0.025 (was 0.075), SQuAD cloze checkpoint 0.175 (was 0.20); loss gap 0.73 vs 1.76 nats. Cloze recall per blank, real / shuffled / zeroed: baseline 0.041 / 0.000 / 0.034, SQuAD cloze 0.188 / 0.008 / 0.034. The old cloze hit counts (13 vs 14 of 40) were inflated by substring matches. Run `scripts/rescore_cloze.py`.
- Answer recall and cloze recall over these 40 paragraphs cover one article only; see [[cloze-eval-paragraphs-one-article]].
