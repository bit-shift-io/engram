Status: done
Complexity: 3 (design 0, scope 1, tests 1, hazards 1)
MCP: none

# Run the pipeline and record results

## What to build
- Run `scripts/fact_qa.py` for real (~7–8 h, background, with a timeout); report the summary.
- Record the outcome in `docs/memory/` (update `reconstruction-keeps-gist-not-facts` or a new entry).

## Files to create/modify
- docs/memory/ (entry + README index)

## Test approach
- Smoke run first; real run only after it passes.

## Acceptance criteria
- [ ] Summary exists with fact gate, length bands and final QA ratio.
- [ ] Memory entry states the result and what it means for the two goals.

## Blocked by
04

## Gotchas
- Long CPU run: launch in the background with `perl -e 'alarm …'`; check progress from the run's metrics file, not by tailing logs.
