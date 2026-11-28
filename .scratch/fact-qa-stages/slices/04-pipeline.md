Status: done
Complexity: 5 (design 1, scope 2, tests 1, hazards 1)
MCP: none

# Unattended two-stage pipeline

## What to build
- `scripts/fact_qa.py`: Stage A → evals (cloze recall, density, val) → fact gate → Stage B → evals (QA, density, cloze recall) → `summary.{json,md}`.
- Resumes after interruption; stops with `gate_failed` when facts are lost; `--smoke` runs it end to end in minutes.

## Files to create/modify
- scripts/fact_qa.py
- tests/test_fact_qa_pipeline.py

## Test approach
- Stub runners: gate pass runs Stage B; gate fail stops before it; length miss still runs Stage B.
- Resume skips finished stages (state file).

## Acceptance criteria
- [ ] One command runs everything unattended and writes the summary.
- [ ] `--smoke` completes on CPU in a few minutes.

## Blocked by
01, 02, 03

## Gotchas
- Follow `scripts/stage3b.py` (gated two-stage pipeline with state.json) and `scripts/squad_cloze.py` (resume, smoke dirs).
