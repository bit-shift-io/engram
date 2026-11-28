---
name: fact-qa-stages-result
description: Stage A (density) + Stage B (answer) from the SQuAD cloze checkpoint kept facts but reached only 0.26× text-QA control F1, and <full> collapsed to terse length; neither goal met
metadata:
  type: context
---

**Why:** 2026-11-28 run of `scripts/fact_qa.py` (Stage A `configs/stage_a_density.yaml` 500 steps → fact gate → Stage B `configs/stage_b_answer.yaml` 500 steps, ~7 h CPU, output `runs/fact_qa/`). It tested whether the SQuAD cloze checkpoint, which stores facts better, would let `<answer>` answer from memory and keep `<full>`/`<terse>` length control.
**How to apply:** Before another attempt at answering from memory or at length control on top of the cloze checkpoint. Storing facts is not the bottleneck; reading them out is.

- Fact gate after Stage A passed: cloze recall 0.133 (needed ≥ 0.13; was 0.168 at the start), cloze gap 2.05 nats. Learning length modes cost some recall; after Stage B it fell to 0.117.
- Length control failed: `<terse>` article ratio 0.115 (in band) but `<full>` 0.148 (band 0.8–1.05) after Stage A and 0.141 after Stage B. `<full>` collapsed to terse length despite control vectors and multi-chunk replay (compare [[prefix-control-token-ignored]]).
- QA (n=200): EM/F1 real 0.105 / 0.176, shuffled 0.025 / 0.068, zeroed 0.000 / 0.010. Memory ÷ control F1 0.256 (control 0.687; goal 0.5). Old Stage 3 lineage reached 0.24×, so the better fact storage barely moved QA.
- Real beats shuffled and zeroed, so answers do use the right memory, but the base model zero-shot without memory scores F1 0.196, above memory QA.
- Stage B in-training QA checks (few questions) sat at F1 0.038 with real = shuffled from step 100 to 500; they were too small to track progress.
- Related: [[reconstruction-keeps-gist-not-facts]] (cloze stores facts, `<full>` does not write them out), [[decoder-ignores-memory]].
