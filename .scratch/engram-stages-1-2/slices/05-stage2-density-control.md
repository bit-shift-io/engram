Status: pending
Complexity: 7 (design 2, scope 2, tests 1, hazards 2)
MCP: none

# Stage 2 density control with `<terse>`

## What to build
- XSum pipeline: truncate articles to 512 tokens, split into up to 4 chunks, encode each, and concatenate the memory.
- Mixed training: batches mix `<full>` reconstruction (WikiText) and `<terse>` summary (XSum) at a configurable ratio.
- Resumes from the Stage 1 checkpoint. The decoder LoRA trains; the encoder continues at a lower LR.

## Files to create/modify
- src/engram/data/compression.py
- src/engram/data/mixed.py
- src/engram/train/trainer.py (per-group LR, init from checkpoint)
- configs/stage2.yaml
- tests/test_compression_data.py

## Test approach
- Data: an article becomes ≤4 chunks and the correct target and control token are attached.
- Mixed sampler: the observed ratio is within tolerance of the config.
- Smoke: Stage 2 runs for 5 steps from a Stage 1 checkpoint.

## Acceptance criteria
- [ ] Stage 2 run finishes in under ~4 hours on the Intel Mac.
- [ ] `<full>` reconstruction accuracy doesn't drop more than 5 points from Stage 1.
- [ ] A `<terse>` render of an XSum article reads as a one-sentence summary.

## Blocked by
04

## Gotchas
- Variable chunk counts → pad the memory and use an attention mask over memory positions.
- Watch for `<terse>` reconstruction loss crowding out `<full>`. Keep at least 50% `<full>` batches initially.
