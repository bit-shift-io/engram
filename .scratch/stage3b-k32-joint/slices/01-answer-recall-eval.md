Status: done
Complexity: 4 (design 1, scope 1, tests 1, hazards 1)
MCP: none

# Answer recall eval

## What to build
- `eval.py --suite recall` reconstructs held-out SQuAD dev paragraphs with `<full>` and reports answer recall and reconstruction token F1.
- Also reports SQuAD `<full>` length ratio (reconstruction ÷ paragraph tokens), to catch Stage 3-style collapse.
- Reproduces the K=16 baseline on the Stage 2 checkpoint (answer recall ≈ 7.5%).

## Files to create/modify
- src/engram/eval/recall.py (new)
- scripts/eval.py
- tests/test_recall_eval.py (new)

## Test approach
- Unit: answer-in-text matching on normalised text, multiple gold answers, empty answers.
- Unit: evaluate_recall on the tiny model returns all metrics in range.
- Manual: Stage 2 checkpoint gives ≈ 0.075 on 40 dev paragraphs.

## Acceptance criteria
- [ ] `eval.py --run <run> --suite recall --n 40` writes `<run>/eval/recall.{json,md}`.
- [ ] K=16 Stage 2 baseline reproduced and noted in the handoff.

## Blocked by
None — can start immediately.

## Gotchas
- Prototype: ../answer_in_memory_prototype.py (copy of the diagnostic). Reuse `normalize`, `f1_score` from engram.eval.qa.
- SQuAD cache is keyed by n: load 200 dev examples and slice, or the loader goes online.
- Stage 3 `<full>` on SQuAD outputs "1999" and stops. That is a real finding, not a bug in this eval.
