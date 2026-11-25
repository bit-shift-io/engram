Status: done
Complexity: 5 (design 1, scope 1, tests 1, hazards 2)
MCP: none

# Text-QA control baseline

## What to build
- Train frozen base + same-size LoRA on SQuAD with `[paragraph][question] → answer` as plain text, same steps and examples as Stage 3's QA share.
- Score it with the slice 02 scorer; cache its result so later runs reuse it.
- Report adds the ratio memory F1 ÷ control F1.

## Files to create/modify
- scripts/train_text_qa.py (new) or a `--text-qa` mode in scripts/train.py
- configs/text_qa.yaml (new)
- src/engram/eval/qa.py, src/engram/eval/report.py
- tests/test_qa_eval.py

## Test approach
- Unit: text-QA example layout masks paragraph and question from loss.
- Smoke: a few steps train; scorer runs on the control model.

## Acceptance criteria
- [ ] Control checkpoint and F1/EM saved under runs/.
- [ ] Report shows the memory ÷ control ratio when a control result exists.

## Blocked by
02

## Gotchas
- Same LoRA rank, LR and step count as the Stage 3 decoder, or the comparison is unfair.
- Run overnight; budget ≤ 4 h.
