# Decisions — cloze eval fix

### Q1: How a cloze hit is counted
**Decision:** Main number = cloze recall: blanked answers found, in any order, in the first N output segments (N = blanks). Report aligned recall alongside.
- **Why:** Memory stores facts; ordering them is the decoder's job. Samples show right facts in wrong slots ("Coldplay" in the CBS slot).
- **Implication:** The N-segment cap stops long guess lists gaming the order-free count.
- **Alternatives considered:** Strict per-position only — mixes storage with placement; first-segment check — ignores most output.

### Q2: Scope of the matcher fix
**Decision:** Fix `answer_in_text` too, so answer recall is re-scored.
- **Why:** It is the same substring matcher; the 0.20 / 0.075 answer recall headline may be inflated.

### Q3: Pass rule after re-scoring
**Decision:** Answer recall ≥ 2× re-scored baseline; real-minus-zeroed loss gap ≥ 2× re-scored baseline gap.
- **Why:** The old 0.15 bar was ~2× the old baseline; fixed numbers were set against an inflated matcher.
- **Alternatives considered:** Keep 0.15 / 1 nat; no pass rule until numbers are seen.

### Q4: Cross-article shuffle
**Decision:** Add the SQuAD article title to dev examples via a one-time download; shuffled memory comes from a different title.
- **Why:** 40 paragraphs span few articles, so an offset pairing can still leak.
- **Implication:** New cache entry; download needs network and the sandbox off.
- **Alternatives considered:** Offset pairing (i + n/2) without network.

### Q5: Short answers
**Decision:** No length cutoff; word-boundary matching only.
- **Why:** Boundaries fix "L" in "League" and "5" in "50"; real one-character answers ("2" sacks) stay valid.

## Assumptions
- The re-scored checkpoints are the only ones compared; no retraining in this feature.
- Paragraph selection for the eval must stay identical to what the trainer excludes (`select_recall_paragraphs`). Blanker changes alter which paragraphs qualify, so the trainer exclusion follows automatically through the shared function.

## Glossary updates
- **Cloze recall** — redefined: order-free share of blanked answers, first N segments, cross-article shuffle.
- **Aligned recall** — new: cloze recall counting only position-matched segments.
