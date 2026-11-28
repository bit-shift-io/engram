Status: done
Complexity: 6 (design 2, scope 1, tests 1, hazards 2)
MCP: none

# Stage A config and loading the SQuAD cloze checkpoint

## What to build
- `configs/stage_a_density.yaml`: Stage 2-style `<full>` (1-chunk and multi-chunk) + `<terse>` mix, control vectors on, plus WikiText and SQuAD cloze replay.
- Initialising from the SQuAD cloze checkpoint works although it lacks control vectors and decoder LoRA: missing modules start at their zero/no-op init.
- A `--smoke` sized run trains a few steps and saves a checkpoint.

## Files to create/modify
- configs/stage_a_density.yaml
- src/engram/train/ (checkpoint init path) — wherever `init_from` loads weights
- tests/test_trainer.py

## Test approach
- Init from a checkpoint saved without control vectors / decoder LoRA: loads, new modules are no-ops (output identical to before at step 0).
- Unexpected missing keys outside those modules still raise.
- Config loads and the mix shares sum as intended.

## Acceptance criteria
- [ ] Stage A smoke run starts from the SQuAD cloze checkpoint and saves a checkpoint.
- [ ] Only control-vector and decoder-LoRA keys may be missing on init.

## Blocked by
None — can start immediately.

## Gotchas
- Read memory `decoder-ignores-memory`: decoder LoRA at low LR (3e-4), encoder side lower; memory read stays on the memory adapter.
- Read memory `prefix-control-token-ignored`: replay `<full>` on SQuAD paragraphs and XSum articles, train `<full>` on multi-chunk memory.
- `configs/stage2.yaml` and `configs/stage3.yaml` carry the LRs and timing notes; keep each stage under ~4 h on CPU.
- Squad cloze replay uses the dev selection exclusion already in the trainer; don't change it.
