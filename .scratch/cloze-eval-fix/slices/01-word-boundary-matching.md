Status: done
Complexity: 3 (design 1, scope 1, tests 0, hazards 1)
MCP: none

# Word-boundary gold-answer matching

## What to build
- One shared matcher: a gold answer matches only where its normalised text starts and ends on a word boundary.
- `answer_in_text` (answer recall) and `make_gold_cloze_example` (blanker) both use it.

## Files to create/modify
- src/engram/eval/recall.py
- src/engram/data/cloze.py
- tests/test_cloze.py, tests/test_recall.py (or wherever `answer_in_text` is tested)

## Test approach
- "L" does not blank "League"; "5" does not blank "50"; "2" blanks "2 sacks".
- Answers with punctuation at the edge ("$5 million", "Levi's Stadium.") still match.
- `answer_in_text(["2"], "2015")` is False; `answer_in_text(["Cam Newton"], "cam newton won")` is True.

## Acceptance criteria
- [ ] No gold answer matches inside a word in the blanker or answer recall.
- [ ] Existing tests still pass.

## Blocked by
None — can start immediately.

## Gotchas
- Answers can start or end with punctuation; `\b` fails next to non-word characters. Use look-around on alphanumerics instead.
- The blanker works on decoded text offsets; keep spans as character offsets into that text.
