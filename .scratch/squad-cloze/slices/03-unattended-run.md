Status: done
Complexity: 4 (design 1, scope 1, tests 1, hazards 1)
MCP: none

# Unattended SQuAD cloze run

## What to build
- `configs/squad_cloze_k16.yaml`: fact K=16 settings, `init_from` the fact K=16 ckpt-3000, 2000 steps, val every 500, cloze 0.2 WikiText + 0.3 SQuAD.
- `scripts/squad_cloze.py`: train → recall eval → cloze recall eval → `runs/squad_cloze/summary.{json,md}` with pass rule and last night's baseline row.
- Early stop at step 1000 when `squad_cloze_val_loss_real` and `_zeroed` differ by < 0.1; summary records the stop reason.

## Files to create/modify
- configs/squad_cloze_k16.yaml
- scripts/squad_cloze.py
- src/engram/eval/report.py
- src/engram/train/trainer.py (early-stop hook)
- tests/test_squad_cloze_pipeline.py

## Test approach
- Stub runners: pass, fail-on-recall, fail-on-gap, early-stop each produce the right summary.
- Smoke run (few steps, tiny data) end to end.

## Acceptance criteria
- [ ] One command runs to a summary without input.
- [ ] Re-running resumes from `state.json` like `scripts/stage3b.py`.
- [ ] Summary flags each pass condition separately.

## Blocked by
01, 02

## Gotchas
- Mix ratios: 0.2 WikiText cloze must be expressed relative to the existing `cloze_ratio` semantics; check the effective shares in a seeded sampler test.
- Estimate wall time on a 50-step timing run before launching; target ≤ 4 h.
