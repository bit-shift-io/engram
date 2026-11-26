Status: done
Complexity: 5 (design 1, scope 2, tests 1, hazards 1)
MCP: none

# K=32 reconstruction warm-up

## What to build
- `configs/stage3b_warmup.yaml`: K=32, Stage 1-style reconstruction, ~3000 steps sized to ~3 h.
- Reconstruction data mixes WikiText chunks with SQuAD train paragraph chunks.
- Validation logs answer recall on a few dev paragraphs.

## Files to create/modify
- configs/stage3b_warmup.yaml (new)
- src/engram/data/reconstruction.py (SQuAD paragraph chunks), src/engram/data/qa.py if needed
- src/engram/train/trainer.py (recall in validation)
- src/engram/config.py
- tests/test_trainer.py, tests/test_chunking.py

## Test approach
- Unit: mixed reconstruction data contains both sources in the configured share.
- Smoke: a few real K=32 steps train from scratch; time one step to set `steps`.

## Acceptance criteria
- [ ] Short smoke run at K=32 completes with finite loss and recall logged.
- [ ] Step time measured; `steps` set so the warm-up fits ~3 h.

## Blocked by
01

## Gotchas
- Never write into or prune the K=16 runs under runs/.
- New data cache keys need network once (HF_HUB_OFFLINE unset, sandbox off). See docs/memory/hf-weights-and-sandbox.md.
- Copy Stage 1 learning rates and schedule from configs/stage1.yaml; only K and data change.
