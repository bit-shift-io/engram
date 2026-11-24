Status: pending
Complexity: 3 (design 1, scope 1, tests 1, hazards 0)
MCP: none

# Density eval and demo

## What to build
- An `eval.py --suite density` mode: mean output length per density for the same memory, plus `<terse>` ROUGE-L on held-out XSum.
- Baseline: the frozen base model summarising from the full article (prompted with "TL;DR:").
- `scripts/demo.py "<text>"` prints the input tokens, memory vector count, compression ratio, and `<full>` and `<terse>` renders.

## Files to create/modify
- src/engram/eval/density.py
- scripts/eval.py
- scripts/demo.py
- tests/test_density_eval.py

## Test approach
- Unit-test the length-ratio and ROUGE aggregation on fixed strings.
- Manual: run the demo on a paragraph pasted into the terminal.

## Acceptance criteria
- [ ] The `<terse>` / `<full>` length ratio is ≤ 0.25 on average.
- [ ] The report puts `<terse>` ROUGE-L next to the base-model baseline.
- [ ] The demo runs in under 30 s on CPU for a 512-token input.

## Blocked by
05
