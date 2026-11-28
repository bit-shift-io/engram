# Fact QA stages

## Problem
- Answering from memory reached only 0.24× the text-QA control's F1 (Stage 3, pre-cloze memory).
- The SQuAD cloze checkpoint stores facts better (cloze recall 0.168 vs 0.056), but never learned `<terse>`, control vectors or multi-chunk `<full>`.
- So neither lineage proves both goals: answers from memory, and different output lengths from the same memory.

## Solution
- Stage A (density): from the SQuAD cloze checkpoint, learn `<full>` and `<terse>` with cloze replay so facts survive.
- A fact gate after Stage A; Stage B (answer) runs only if facts were kept.
- Stage B: Stage 3 `<answer>` training from Stage A, judged against the text-QA control, with the regression gate.

## Out of Scope
- Making `<full>` write out more facts. Answer recall in `<full>` is not a goal.
- K=32, idea slots, new architectures.
- Changing the 40-paragraph eval selection or the text-QA control.

## Cross-cutting Constraints
- Run settings: `.venv/bin/python`, `HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1`; data caches are keyed by sample count.
- Facts outrank length control: a lost fact gate stops the pipeline; a missed length band is reported and Stage B still runs.
- Memory is read through the memory-only adapter; any decoder LoRA trains at low LR (see memory `decoder-ignores-memory`).
- Every mode is replayed on every input type it is evaluated on (see memory `prefix-control-token-ignored`).
- CPU only: each stage must fit in ~4 h.

## Acceptance Criteria
- [ ] One command runs Stage A → fact gate → Stage B unattended, resumes after interruption, and writes a summary.
- [ ] Fact gate: cloze recall (real) ≥ 0.13 and real-minus-zeroed cloze loss gap ≥ 1.5 nats; failure stops before Stage B.
- [ ] Length bands (Stage 2 regression bands, `<full>` gap ≥ 0.9) reported after Stage A and Stage B; a miss warns, never stops.
- [ ] Final: memory QA F1 ÷ text-QA control F1 reported against 0.5, with shuffled and zeroed QA F1 beside it.
- [ ] Results recorded in `docs/memory/`.

## References
- Glossary: Cloze recall, Answer task, Text-QA control, Regression gate, Fact gate, Density stage (`docs/CONTEXT.md`).
- Memory: `reconstruction-keeps-gist-not-facts`, `decoder-ignores-memory`, `prefix-control-token-ignored`, `cloze-eval-paragraphs-one-article`.
- Start checkpoint: `runs/squad_cloze/train/20261127-163240`. Stage 3 recipe: `configs/stage3.yaml`. Gated pipeline pattern: `scripts/stage3b.py`.
