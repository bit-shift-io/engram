Status: in-progress
Complexity: 5 (design 1, scope 1, tests 1, hazards 2)
MCP: none

# Stage 3 run with regression gates

## What to build
- `configs/stage3.yaml`: 50/25/25 mix from the Stage 2 checkpoint, budget ≤ 4 h.
- Every validation logs QA F1 (real/shuffled/zeroed), `<full>` gap, and density metrics.
- Final report flags regressions against Stage 2 and recommends K=32 when the ratio < 0.5.

## Files to create/modify
- configs/stage3.yaml (new)
- src/engram/train/trainer.py (QA validation hook)
- src/engram/eval/report.py
- docs/training.md (record Stage 3 results), docs/CONTEXT.md
- tests/test_trainer.py

## Test approach
- Unit: regression flag and K=32 recommendation logic on fixed numbers.
- Smoke config run exercises the validation hook.

## Acceptance criteria
- [ ] Overnight run completes; report meets the README acceptance criteria.
- [ ] Results and lessons recorded in docs/training.md and docs/memory/.

## Blocked by
01, 02, 03

## Gotchas
- Time the step first: QA micro-batches with up to 4 chunks cost ~4 s each; set steps to fit 4 h.
- Never touch the accepted Stage 1/2 runs when pruning runs/.
