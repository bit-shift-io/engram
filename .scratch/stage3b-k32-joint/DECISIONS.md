# Decisions — stage3b-k32-joint

### Q1: How to get K=32 memory
**Decision:** Train a K=32 encoder from scratch with a full reconstruction warm-up (~3000 steps, ~3 h).
- **Why:** Cleanest capacity test; answer recall after warm-up is a result on its own.
- **Implication:** Pipeline runs ~7.5 h; user approved.
- **Alternatives considered:** Duplicate the K=16 memory queries (faster, but confounds a failure); cold joint start (cold-encoder risk).

### Q2: Go/no-go between stages
**Decision:** Gate on answer recall ≥ 25% over 40 SQuAD dev paragraphs (K=16 baseline 7.5%).
- **Why:** If memory does not hold answers, the joint stage cannot fix it.
- **Implication:** Summary printed and saved first; pass continues with no human pause.
- **Alternatives considered:** Always run both stages — wastes night 2 on a known failure.

### Q3: Warm-up data
**Decision:** Reconstruction on WikiText plus SQuAD train paragraphs.
- **Why:** The gate measures SQuAD-style text; the encoder must see it.
- **Alternatives considered:** WikiText only, as Stage 1.

### Q4: Joint mix
**Decision:** 40% `<answer>`, 35% `<full>` (half SQuAD paragraphs, half WikiText/XSum), 25% `<terse>` XSum, all from the warm-up checkpoint.
- **Why:** Stage 3 `<full>` collapsed on SQuAD when only `<answer>` saw SQuAD.
- **Alternatives considered:** Stage 3's 50/25/25 with no SQuAD `<full>`.

### Q5: Text-QA control
**Decision:** Re-train inside the pipeline with steps = `<answer>` share × joint steps.
- **Why:** Keeps the ratio fair; the cached Stage 3 control used 320 steps.

## Assumptions
- K=32 per-step cost is close to K=16 (memory is a small share of decoder length). Slice 02 times it.
- The Stage 3 code (answer task, QA eval, control, gates) works at K=32 unchanged.

## Trade-offs
- One more night of compute for an unconfounded result.

## New CONTEXT.md terms
- **Answer recall** — share of held-out SQuAD paragraphs whose `<full>` reconstruction from memory contains a gold answer.
- **Recall gate** — answer recall ≥ 25% after warm-up; decides whether the joint stage runs.
