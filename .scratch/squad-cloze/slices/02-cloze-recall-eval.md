Status: done
Complexity: 3 (design 1, scope 1, tests 1, hazards 0)
MCP: none

# Cloze recall eval suite

## What to build
- `scripts/eval.py --suite cloze`: 40 distinct SQuAD dev paragraphs whose first chunk contains the gold answer; blank it; decode in cloze layout with real, shuffled and zeroed memory.
- Report hit rate (answer in the first ` | ` segment) and mean teacher-forced answer loss per condition; write `<run>/eval/cloze_recall.{json,md}`.

## Files to create/modify
- src/engram/eval/cloze_recall.py
- scripts/eval.py
- tests/test_cloze.py

## Test approach
- Paragraph selection: duplicates and answer-not-in-chunk are skipped; stops at n.
- Stub decoder: hit rate and loss averaging per condition.

## Acceptance criteria
- [x] Re-baselined (user decision 2026-11-27): fact K=16 ckpt-3000 under this eval = real 3.48 / shuffled 4.60 / zeroed 4.21 nats; hits 5 / 3 / 11 of 40. Old 5.0/5.5/6.6 numbers came from a lost scratch script and are not comparable.

## Blocked by
None — can start immediately.

## Gotchas
- Reference implementation (temporary): the session scratchpad script `cloze_recall.py`; logic summarised above, do not import it.
- Load 3000 dev examples (cached) to reach 40 distinct paragraphs; 200 gives only 10.
