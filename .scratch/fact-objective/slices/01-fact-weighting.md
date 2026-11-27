Status: done
Complexity: 5 (design 1, scope 1, tests 1, hazards 2)
MCP: none

# Fact-weighted reconstruction loss

## What to build
- Config `fact_weighting: bool = False`, `fact_weight_cap: float = 5.0`.
- When on, reconstruction loss weights each target token by the frozen decoder's loss on it with zeroed memory.
- Weights: detached, normalised to mean 1 per chunk over non-pad tokens, clipped at the cap, renormalised.

## Files to create/modify
- src/engram/model/decoder.py (per-token loss, weighted variant)
- src/engram/train/trainer.py
- src/engram/config.py
- tests/test_decoder.py or tests/test_trainer.py

## Test approach
- Unit: weights mean ≈ 1, ≤ cap, no grad through weights, pad tokens weight 0.
- Unit: a token the zeroed-memory decoder already predicts gets lower weight than a surprising one (tiny model, constructed case).
- Unit: `fact_weighting: false` gives the identical loss to before.
- Smoke: a few tiny-model steps train with finite loss.

## Acceptance criteria
- [ ] Weighted loss used when the flag is on; default behaviour unchanged.
- [ ] Extra cost measured on a real K=16 step and noted in the handoff.

## Blocked by
None — can start immediately.

## Gotchas
- "Zeroed memory" must go through the same path the memory-necessity eval uses (zeros of the memory shape), not by dropping memory positions.
- The decoder is frozen but the memory adapter is not: compute the weight pass under `torch.no_grad()`.
- See docs/memory/peft-multi-adapter-grads.md before touching adapters.
