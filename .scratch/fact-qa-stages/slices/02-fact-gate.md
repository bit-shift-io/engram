Status: done
Complexity: 3 (design 1, scope 1, tests 0, hazards 1)
MCP: none

# Fact gate after Stage A

## What to build
- `fact_gate(cloze, density, val)` returns pass/fail for facts (cloze recall real ≥ 0.13, cloze loss gap ≥ 1.5) and separate warnings for each length band.
- Markdown names each check, its value and threshold.

## Files to create/modify
- src/engram/eval/report.py
- tests/test_report.py (or the existing report tests)

## Test approach
- Facts pass + length miss → passed, with a warning listed.
- Facts fail (either check) → failed, regardless of length.
- Thresholds at exactly the boundary pass.

## Acceptance criteria
- [ ] Gate result separates `facts_passed` from `length_warnings`.
- [ ] Length bands reuse the existing Stage 2 regression bands (`stage3_gates` logic), not copies.

## Blocked by
None — can start immediately.
