Status: done
Complexity: 6 (design 2, scope 1, tests 1, hazards 2)
MCP: none

# Cloze task from memory

## What to build
- `find_fact_spans(text) -> list[(start, end)]`: digit runs, years, month names, capitalised words not sentence-initial.
- Cloze example: encoder sees the original chunk. Decoder prompt is the chunk with each fact span replaced by `___`, then a separator; target is the fact spans in order, joined by ` | `. Loss on target tokens only.
- Config `cloze_ratio: float = 0.0`; trainer draws a cloze batch with that probability, else reconstruction. Chunks with no fact spans fall back to reconstruction.

## Files to create/modify
- src/engram/data/cloze.py (new)
- src/engram/train/trainer.py
- src/engram/config.py
- tests/test_cloze.py (new)

## Test approach
- Unit: span finder on fixed sentences (numbers, "March 1999", "Notre Dame", sentence-initial "The" excluded).
- Unit: cloze example builder — blanked prompt contains no fact text; target equals spans in order; loss mask covers target only.
- Unit: sampler share ≈ cloze_ratio over many draws; `cloze_ratio: 0` unchanged.
- Smoke: tiny-model steps with cloze on, finite loss; validation logs `cloze_val_loss` with real / shuffled / zeroed memory.

## Acceptance criteria
- [ ] Cloze batches train end to end; default behaviour unchanged.
- [ ] Validation logs cloze loss for real, shuffled and zeroed memory.

## Blocked by
None — can start immediately.

## Gotchas
- Work on decoded text, then re-tokenise; token boundaries do not align with spans. Truncate prompt + target to the decoder length budget.
- Control vectors are off in Stage 1 settings; the cloze prompt itself tells the decoder the task. Do not add a control token.
- The blanked text is visible to the decoder, so facts can only come from memory. Keep it that way: never leak fact text into the prompt.
