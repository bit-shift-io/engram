Status: done
Complexity: 5 (design 1, scope 2, tests 1, hazards 1)
MCP: none

# SQuAD gold-answer cloze in training

## What to build
- Build cloze examples from SQuAD train paragraphs: first chunk, EOS stripped, every gold answer inside it blanked; target is the answers in text order, ` | `-joined.
- New config `squad_cloze_ratio`: share of batches drawn as SQuAD cloze, taken from the non-cloze `<full>` share. 0 leaves batches unchanged.
- Validation logs `squad_cloze_val_loss_{real,shuffled,zeroed}` on SQuAD dev paragraphs not used by the recall evals' first 40.

## Files to create/modify
- src/engram/data/cloze.py (gold-span example builder, paragraph grouping)
- src/engram/train/trainer.py (sampler + val logging)
- src/engram/config.py
- tests/test_cloze.py, tests/test_trainer.py

## Test approach
- Paragraph with two answers → both blanked, target order matches text order, no EOS in the question.
- Answer outside the first chunk → dropped; no answers left → paragraph skipped.
- `squad_cloze_ratio=0` → sampler draws identical batches to today (seeded).
- Tiny run logs the three new val keys.

## Acceptance criteria
- [ ] SQuAD only enters training via cloze batches.
- [ ] Dev paragraphs never enter training.
- [ ] Existing 90 tests still pass.

## Blocked by
None — can start immediately.

## Gotchas
- Group questions by paragraph text; SQuAD repeats a paragraph once per question.
- `make_cloze_example` uses regex spans; reuse its truncation and layout, not its span finder.
- A val batch of 1 cannot be shuffled; trim like `cloze_val`.
- Data caches are keyed by sample count; new counts trigger a download (needs network, not offline mode).
