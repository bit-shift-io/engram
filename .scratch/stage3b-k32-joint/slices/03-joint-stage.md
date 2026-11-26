Status: pending
Complexity: 6 (design 1, scope 2, tests 1, hazards 2)
MCP: none

# Joint `<full>`/`<terse>`/`<answer>` stage

## What to build
- `configs/stage3b_joint.yaml`: from the warm-up checkpoint, mix 40% `<answer>`, 35% `<full>` (half SQuAD paragraphs), 25% `<terse>` XSum.
- `MixedSampler` gains a SQuAD-paragraph `<full>` source.
- Regression gates add SQuAD `<full>` length ratio and answer recall; `configs/text_qa_3b.yaml` matches the `<answer>` step count.

## Files to create/modify
- configs/stage3b_joint.yaml (new), configs/text_qa_3b.yaml (new)
- src/engram/data/mixed.py
- src/engram/train/trainer.py
- src/engram/eval/report.py (new gates)
- src/engram/config.py
- tests/test_trainer.py

## Test approach
- Unit: sampler draws SQuAD `<full>` batches at the configured share.
- Unit: new gates flag a collapsed SQuAD `<full>` (short output) on fixed numbers.
- Smoke: a few joint steps from a K=32 checkpoint; validation logs QA and new gates.

## Acceptance criteria
- [ ] Joint smoke run completes; gates include SQuAD `<full>`.
- [ ] Step time measured; `steps` set so the joint stage fits ~3.5 h.

## Blocked by
01, 02

## Gotchas
- All three control vectors start at zero together; no Stage 2 expand-on-load path here.
- Stage 3 gate bands for XSum were chosen without review (`STAGE2_GATES`); keep them, add SQuAD bands beside them.
