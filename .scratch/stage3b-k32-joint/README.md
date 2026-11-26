# Stage 3b: K=32 joint training

## Problem
- Stage 3 QA from memory failed: memory F1 0.051, shuffled 0.048, text-QA control 0.687.
- K=16 memory keeps the answer in `<full>` reconstruction only 7.5% of the time. It holds gist, not specifics.
- Training `<answer>` after `<full>` made `<full>` collapse on SQuAD paragraphs. The XSum-only gates missed it.

## Solution
- One unattended pipeline: K=32 reconstruction warm-up, an answer recall gate, then a joint `<full>`/`<terse>`/`<answer>` stage.
- The gate prints a findings summary. Pass (answer recall ≥ 25%) continues straight to the joint stage; fail stops.
- New regression gates check `<full>` on SQuAD paragraphs, not only XSum.

## Out of Scope
- Initialising K=32 from the K=16 encoder (duplicated memory queries).
- K=64, new encoder objectives. Revisit only if the gate fails.
- T1 idea slots, T2, latent reasoner.

## Cross-cutting Constraints
- K = 32 everywhere in this feature. Never load or overwrite the accepted K=16 runs.
- Every evaluation runs real / shuffled / zeroed memory through `Decoder.loss` / `Decoder.generate`. See docs/memory/memory-necessity-eval.md.
- Every output mode trains on every input type it is evaluated on. See docs/memory/prefix-control-token-ignored.md.
- Decoder LoRA 3e-4; encoder side at the Stage 2 scale. See docs/memory/decoder-ignores-memory.md.
- SQuAD paragraphs over 4 chunks are skipped. Gate and QA eval use SQuAD dev; training uses SQuAD train.
- Whole pipeline ≤ ~7.5 h unattended. Each stage resumes from its last checkpoint.

## Acceptance Criteria
- [ ] Pipeline runs warm-up → gate → summary → joint stage → control → final evals with no human input.
- [ ] Gate summary printed and saved before the joint stage starts; a failed gate stops the pipeline.
- [ ] Final report: memory QA EM/F1 real/shuffled/zeroed, control F1, ratio, answer recall, all regression gates incl. SQuAD `<full>`.
- [ ] Real memory F1 clearly beats shuffled.
- [ ] Results recorded in docs/training.md and docs/memory/.
- [ ] Fast tests pass.

## References
- docs/training.md (Stage 3 results), docs/architecture.md
- docs/memory/prefix-control-token-ignored.md, decoder-ignores-memory.md, memory-necessity-eval.md
- Stage 3 plan: .scratch/stage3-qa/
