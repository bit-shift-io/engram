Status: done
Complexity: 5 (design 1, scope 2, tests 1, hazards 1)
MCP: none

# QA evaluation from memory

## What to build
- `scripts/eval.py --qa` scores a checkpoint on a fixed held-out SQuAD dev subset: EM/F1 with real, shuffled and zeroed memory.
- Also scores the zero-shot base model with paragraph + question in context.
- Writes results into the run report next to the existing ablation and density tables.

## Files to create/modify
- src/engram/eval/qa.py (new): normalise, EM, F1, evaluate_qa
- src/engram/eval/report.py
- scripts/eval.py
- tests/test_qa_eval.py (new)

## Test approach
- Unit: SQuAD normalisation, EM/F1 against multiple gold answers, known examples.
- Unit: evaluate_qa on a tiny model runs all three memory conditions.

## Acceptance criteria
- [ ] Report shows memory EM/F1 for real / shuffled / zeroed, plus zero-shot base F1.
- [ ] Held-out set fixed by seed and size in config.

## Blocked by
01

## Gotchas
- Generate through `Decoder.generate` so `_mem` and `_flags` hooks are on (docs/memory/memory-necessity-eval.md).
- Stop at EOS or newline; cap ~32 tokens.
