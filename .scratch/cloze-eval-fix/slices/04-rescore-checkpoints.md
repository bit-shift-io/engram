Status: done
Complexity: 4 (design 1, scope 1, tests 1, hazards 1)
MCP: none

# Re-score both checkpoints with a relative pass rule

## What to build
- `scripts/rescore_cloze.py`: runs answer recall and cloze recall on the baseline and SQuAD cloze checkpoints, writes `runs/cloze_rescore/summary.{json,md}`.
- Pass rule: answer recall ≥ 2× baseline, and real-minus-zeroed loss gap ≥ 2× baseline gap; each flagged separately.
- Summary shows old vs re-scored numbers for both checkpoints.

## Files to create/modify
- scripts/rescore_cloze.py
- src/engram/eval/report.py
- tests/test_squad_cloze_pipeline.py (or a new rescore test)

## Test approach
- Stub runners: pass, fail-on-recall, fail-on-gap each produce the right flags.
- Baseline of 0 answer recall → rule reported as undefined, not a pass.

## Acceptance criteria
- [ ] One command re-scores both checkpoints without input.
- [ ] Summary flags each pass condition separately.

## Blocked by
01, 02, 03

## Gotchas
- Write new reports to a new directory; don't overwrite the original `eval/` reports in either run, so old numbers stay comparable.
- Full re-score takes ~25 min; run it in the background with a timeout.
