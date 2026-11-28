Status: done
Complexity: 4 (design 1, scope 1, tests 1, hazards 1)
MCP: none

# Stage B config and final QA result

## What to build
- `configs/stage_b_answer.yaml`: Stage 3 mix (`<answer>`, `<full>` and `<terse>` replay) plus cloze replay, from a Stage A checkpoint.
- Final summary: memory QA F1 and EM with real / shuffled / zeroed memory, ratio to the text-QA control vs 0.5, regression gate, and cloze recall after Stage B.

## Files to create/modify
- configs/stage_b_answer.yaml
- src/engram/eval/report.py
- tests/test_report.py

## Test approach
- Ratio ≥ 0.5 → pass; below → fail; missing control → reported undefined, not pass.
- Regression gate misses appear as warnings beside the QA result.

## Acceptance criteria
- [ ] Final summary shows QA F1 per memory condition and the control ratio.
- [ ] Shuffled and zeroed QA F1 are shown, so a pass with weak memory dependence is visible.

## Blocked by
01

## Gotchas
- Text-QA control result lives at `runs/text_qa_control.json` (F1 0.69, n=200); evaluate memory QA on the same n.
