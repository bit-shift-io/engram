# Decisions — fact-objective

### Q1: Which K first
**Decision:** Fresh K=16 first; fresh K=32 (4000 steps) only if K=16 fails.
- **Why:** K=16 trains faster with no plateau; K=16 memory works, so facts are untested, not ruled out.
- **Alternatives considered:** Continue the K=32 WikiText-only checkpoint (needs a matched control run).

### Q2: Fresh vs continued training
**Decision:** Fresh, with Stage 1 settings; only the objective changes.
- **Why:** The existing K=16 Stage 1 run is then the control. No extra control run.

### Q3: Objective mix
**Decision:** 70% fact-weighted reconstruction, 30% cloze.
- **Why:** Weighting keeps the gist objective but aims it at facts; cloze trains fact recall directly.
- **Implication:** Weights normalised to mean 1 per chunk and capped, so a few tokens cannot dominate.

### Q4: Fact detection
**Decision:** Regex: digit runs, month names/years, capitalised words not sentence-initial.
- **Why:** No new dependency; good enough for SQuAD-style answers.

### Q5: Pass criterion
**Decision:** Answer recall ≥ 0.15 on 40 dev paragraphs and real > shuffled ROUGE-L.
- **Why:** Double the 0.075 baseline; a clear signal, below the Stage 3b gate of 0.25.

### Q6: Supervision
**Decision:** Build, then launch the run without review stops.
- **Why:** User away overnight.

## Assumptions
- Fact weighting costs one extra frozen-decoder forward per batch (zeroed memory, no grad); ~1.3× step time.
- Pre-computing weights once per chunk is equivalent, since the decoder is frozen. Allowed as an optimisation.

## New CONTEXT.md terms
- **Fact token** — a chunk token inside a regex-detected name, number or date span.
- **Fact weighting** — per-token reconstruction loss weight from the frozen decoder's zeroed-memory loss.
- **Cloze task** — decoder sees the chunk with fact spans blanked and must output the fact spans, from memory.
