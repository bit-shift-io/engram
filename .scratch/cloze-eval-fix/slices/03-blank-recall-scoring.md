Status: done
Complexity: 3 (design 1, scope 1, tests 1, hazards 0)
MCP: none

# Cloze recall and aligned recall scoring

## What to build
- Each cloze example keeps its blanks in order, each with its accepted answer strings (merged variants form one blank).
- Cloze recall = blanks whose answers appear in the first N ` | ` segments, any order, each segment used once; aligned recall = segment k matches blank k.
- Report both per condition in `cloze_recall.{json,md}`, beside answer loss; drop the old hit count.

## Files to create/modify
- src/engram/data/cloze.py (keep per-blank answers)
- src/engram/eval/cloze_recall.py
- tests/test_cloze.py

## Test approach
- Stub decoder output in the wrong order → full cloze recall, zero aligned recall.
- Output longer than N segments → extra segments ignored.
- One segment cannot satisfy two blanks.

## Acceptance criteria
- [ ] JSON and markdown report cloze recall and aligned recall per condition.
- [ ] Matching uses slice 01's word-boundary matcher.

## Blocked by
01
