Status: pending
Complexity: 6 (design 2, scope 1, tests 2, hazards 1)
MCP: none

# Unattended pipeline

## What to build
- `scripts/stage3b.py`: warm-up → recall gate → print and save summary → joint stage → text-QA control → final QA, recall and density evals → final summary.
- Recall gate below 25% stops after the summary with a clear reason.
- Re-running resumes from the last completed stage.

## Files to create/modify
- scripts/stage3b.py (new)
- src/engram/eval/report.py (gate and final summaries)
- tests/test_pipeline.py (new)
- docs/training.md (Stage 3b how-to-run)

## Test approach
- Unit: gate pass continues, gate fail stops; stage resume skips finished stages (stub the stage runners).
- Smoke: whole pipeline with tiny step counts end to end.

## Acceptance criteria
- [ ] Smoke pipeline runs all stages unattended and writes both summaries.
- [ ] Forced gate failure stops before the joint stage.
- [ ] Launch command recorded in docs/training.md.

## Blocked by
01, 02, 03

## Gotchas
- Write the gate summary to the run folder and stdout before starting the joint stage.
- Stage state must live on disk (e.g. `<pipeline dir>/state.json`) so a pause can resume.
