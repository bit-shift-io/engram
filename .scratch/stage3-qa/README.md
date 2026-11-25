# Stage 3: QA from memory

## Problem
- Stage 2 memory holds gist; names, numbers and facts come out wrong.
- We cannot yet test whether memory holds usable meaning, not just surface text.
- T1 idea slots (Stage 4) train on per-slot QA and fold from T0, so they need a working T0 QA objective first.

## Solution
- Add an `<answer>` task: `[memory][<answer>][question] → answer`, trained on SQuAD v1.1.
- Score answers by SQuAD EM/F1 against a fair text-QA control that reads the paragraph instead of memory.
- Keep Stage 2 skills alive with a balanced mix and regression gates.

## Out of Scope
- Synthetic QA from a larger LLM. Revisit only if SQuAD works.
- K=32. Fallback only (see Acceptance Criteria).
- T1 idea slots, T2 persistence, latent reasoner.

## Cross-cutting Constraints
- K stays 16. Train from the Stage 2 checkpoint `runs/20261125-163535/ckpt-600.pt`.
- Mix: 50% `<answer>`, 25% `<full>` replay (single-chunk and whole-article), 25% `<terse>` XSum replay.
- Decoder LoRA stays at 3e-4; encoder side at the Stage 2 scale. See docs/memory/decoder-ignores-memory.md.
- Every evaluation runs real / shuffled / zeroed memory through `Decoder.loss` / `Decoder.generate`. See docs/memory/memory-necessity-eval.md.
- SQuAD paragraphs over 4 chunks (512 tokens) are skipped.
- CPU budget: a full run must fit an overnight window (≤ 4 h).

## Acceptance Criteria
- [ ] Memory QA F1 and EM reported on a fixed held-out SQuAD dev set, with shuffled and zeroed memory alongside.
- [ ] Real memory F1 clearly beats shuffled and zeroed memory.
- [ ] Text-QA control F1 and zero-shot base-model F1 reported on the same set.
- [ ] Ratio memory F1 ÷ text-QA control F1 reported. Target ≈ 1.
- [ ] If the ratio < 0.5, the report recommends a K=32 arm.
- [ ] No regression from Stage 2: `<full>` gap ≥ ~1.0 nats; `<terse>`/`<full>` length ratios and `<terse>` ROUGE-L near 0.10 / 0.91 / 0.184.
- [ ] Fast tests pass.

## References
- docs/training.md (Stage 3), docs/architecture.md (memory tiers), docs/implementation-plan.md
- docs/memory/prefix-control-token-ignored.md, decoder-ignores-memory.md, memory-necessity-eval.md
- Stage 1–2 plan: .scratch/engram-stages-1-2/
