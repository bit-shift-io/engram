Status: done
Complexity: 4 (design 1, scope 1, tests 1, hazards 1)
MCP: none

# Reconstruction and memory-ablation eval

## What to build
- `scripts/eval.py --run runs/<id> --suite reconstruction` reports greedy reconstruction token accuracy and ROUGE-L on held-out chunks.
- The same suite runs under three memory conditions: real, zeroed, and shuffled (memory from a different chunk).
- Writes `eval/reconstruction.json` and a markdown table to the run directory.

## Files to create/modify
- src/engram/eval/reconstruct.py
- src/engram/eval/ablation.py
- src/engram/eval/report.py
- scripts/eval.py
- tests/test_eval.py

## Test approach
- Unit-test the token-accuracy and ablation transforms on fixed tensors.
- Integration: on the overfit checkpoint from slice 02, real memory scores much higher than zeroed memory.

## Acceptance criteria
- [ ] The report shows the real/zeroed/shuffled accuracy and the compression ratio.
- [ ] After the Stage 1 run, real ≥ 80% and zeroed/shuffled near baseline (K=16).
- [ ] Results for the K sweep {4, 8, 16, 32} are recorded in one comparison table.

## Blocked by
03

## Gotchas
- Shuffled memory must come from a *different* chunk in the batch. Derangement, not permutation.
