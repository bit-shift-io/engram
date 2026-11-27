# Decisions

### Q1: Start point
**Decision:** Continue from the fact K=16 checkpoint (`init_from`, fresh optimizer and step).
- **Why:** It already keeps the gist (recon F1 0.31) and a faint fact trace; 3.5 h vs 6 h.
- **Implication:** Results mix last night's training with this run's.
- **Alternatives considered:** Fresh 3000-step run — deferred until this shows SQuAD cloze moves recall.

### Q2: What to blank
**Decision:** Gold answer spans of all SQuAD train questions on the paragraph.
- **Why:** Trains exactly what answer recall measures.
- **Alternatives considered:** Regex fact spans (looser match); both (too many blanks per paragraph).

### Q3: Pass rule
**Decision:** Answer recall ≥ 0.15 through `<full>`, and cloze recall real beats zeroed by ≥ 1 nat answer loss; reported separately.
- **Why:** Separates "facts stored" from "`<full>` writes them out". Last night's gap was 0.5 nats.

### Q4: Mix and length
**Decision:** 50% fact-weighted WikiText reconstruction, 20% WikiText cloze, 30% SQuAD cloze; 2000 steps; val every 500.
- **Why:** Keeping WikiText cloze and reconstruction protects the gist; SQuAD only via cloze avoids the mix stall.

### Q5: Early stop
**Decision:** Stop at step 1000 when SQuAD cloze val real and zeroed loss differ by < 0.1 nats.
- **Why:** K=32 runs wasted hours with memory unused; this catches it at a third of the run.

## Assumptions
- SQuAD train paragraphs fit in one 128-token chunk often enough; longer ones are truncated to the first chunk, keeping only answers inside it.
- 40 distinct dev paragraphs come from the first 3000 dev questions (cache already built).

## Cloze recall eval re-baselined (2026-11-27, during implementation)
The new eval did not reproduce last night's fact K=16 numbers (real 5.0 / zeroed 5.5 / shuffled 6.6, 2/40). The reference script was lost, and its blanking, hit and loss-averaging choices are unknown. The user chose to accept the new eval as the definition and re-baseline: K=16 ckpt-3000 = real 3.48 / shuffled 4.60 / zeroed 4.21 nats, hits 5 / 3 / 11 of 40. The summary compares against these, not the old numbers. All gold answers in the paragraph are blanked; a hit = any gold answer in the first ` | ` segment; loss = per-example mean.
