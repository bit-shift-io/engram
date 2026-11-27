Status: done
Complexity: 5 (design 1, scope 1, tests 1, hazards 2)
MCP: none

# Cross-article shuffled memory

## What to build
- SQuAD QA examples carry the article `title`.
- Cloze recall's shuffled condition (generation and answer loss) pairs each paragraph with memory from a different title.

## Files to create/modify
- src/engram/data/qa.py
- src/engram/eval/cloze_recall.py
- tests/test_cloze.py

## Test approach
- Paragraphs from 2+ titles: every shuffled pairing crosses titles.
- All paragraphs from one title → clear error, not a silent same-article pairing.
- Answer-loss shuffled batches also cross titles.

## Acceptance criteria
- [ ] Shuffled memory always comes from a different article.
- [ ] The 3000-example dev cache with titles exists locally after one download.

## Blocked by
None — can start immediately.

## Gotchas
- Adding `title` changes nothing in the cache key; old caches lack the field. Bump the key (e.g. `-ans-t`) so a new cache is built, and keep loaders tolerant of the old one elsewhere.
- The download needs network and the sandbox off (`docs/memory/hf-weights-and-sandbox.md`). The implementer cannot do this offline: stop and report the exact command for the user to run.
- 40 selected paragraphs may span only 2–3 articles; check the shuffle still has a partner for each.
