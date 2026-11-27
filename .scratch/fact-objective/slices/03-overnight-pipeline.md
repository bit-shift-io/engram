Status: done
Complexity: 6 (design 1, scope 2, tests 2, hazards 1)
MCP: none

# Unattended K=16 → K=32 run

## What to build
- `configs/fact_k16.yaml`: Stage 1 settings, K=16, 3000 steps, `fact_weighting: true`, `cloze_ratio: 0.3`, recall logged at validation.
- `configs/fact_k32.yaml`: same, K=32, 4000 steps.
- `scripts/fact_objective.py`: K=16 train → recall (n=40) + memory-use eval → pass? stop : K=32 train → same evals → summary (`<dir>/summary.{json,md}`). State on disk; re-run resumes.

## Files to create/modify
- configs/fact_k16.yaml, configs/fact_k32.yaml (new)
- scripts/fact_objective.py (new)
- src/engram/eval/report.py (pass check, summary)
- tests/test_fact_pipeline.py (new)
- docs/training.md (how to run)

## Test approach
- Unit (stubbed runners): pass stops after K=16; fail runs K=32; resume skips finished stages.
- Smoke: whole pipeline with tiny step counts under runs_smoke_fact/.

## Acceptance criteria
- [ ] Smoke pipeline runs both arms (forced fail) and writes the summary.
- [ ] Summary shows each arm against baselines: K=16 Stage 1 and K=32 WikiText-only (recall 0.050).
- [ ] Step times measured; total estimate in the handoff.

## Blocked by
01, 02

## Gotchas
- Pass = answer_recall ≥ 0.15 and real ROUGE-L > shuffled ROUGE-L on WikiText val.
- Memory-use eval: `evaluate_reconstruction` on 15 WikiText val chunks and 15 SQuAD dev paragraph chunks (only 15 distinct exist in the cached 200). Write results only into the new run dirs.
- Reuse scripts/stage3b.py patterns (runner dict, state.json, in-process `train.main`).
- Data caches are keyed by sample count; keep cached sizes (docs/memory/hf-weights-and-sandbox.md).
