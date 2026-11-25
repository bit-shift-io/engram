Status: done
Complexity: 7 (design 2, scope 2, tests 1, hazards 2)
MCP: none

# `<answer>` task trains end to end

## What to build
- Load SQuAD v1.1, chunk each paragraph (≤ 4 chunks, else skip), build `{chunks, question_ids, answer_ids}` examples.
- Add `<answer>` control token and flag vector; decoder input `[memory][<answer>][question][answer]`, loss on answer only.
- `MixedSampler` gains an `answer` share; `train.py` trains a smoke run from the Stage 2 checkpoint.

## Files to create/modify
- src/engram/tokens.py
- src/engram/data/qa.py (new)
- src/engram/data/mixed.py
- src/engram/model/decoder.py (question prompt segment excluded from loss; flags cover it)
- src/engram/model/engram.py, src/engram/train/trainer.py
- src/engram/config.py, configs/smoke.yaml
- tests/test_qa_data.py (new), tests/test_control_vectors.py, tests/test_trainer.py

## Test approach
- Unit: example building, long-paragraph skip, answer-only loss mask (question positions are -100).
- Unit: loading a Stage 2-shaped checkpoint into a 3-token model keeps `<full>`/`<terse>` vectors and zeros `<answer>`.
- Smoke: a few mixed steps run and loss is finite.

## Acceptance criteria
- [ ] Mixed batches of answer / full / terse train without error from the Stage 2 checkpoint.
- [ ] Stage 2 outputs unchanged at step 0 (zero `<answer>` vector).
- [ ] Fast tests pass.

## Blocked by
None — can start immediately.

## Gotchas
- Adding a token resizes embeddings and grows `ControlVectors.vec` and the control embedding: Stage 2 checkpoints will not load strictly. Write an explicit expand-on-load.
- Keep `CONTROL_TOKENS` order (`<full>`, `<terse>`, then `<answer>`) so existing ids don't shift.
- SQuAD download needs network and sandbox off (docs/memory/hf-weights-and-sandbox.md).
- Use `use_adapter()`, not `set_adapter` (docs/memory/peft-multi-adapter-grads.md).
