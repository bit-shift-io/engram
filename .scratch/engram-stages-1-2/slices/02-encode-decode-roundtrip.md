Status: pending
Complexity: 7 (design 2, scope 2, tests 1, hazards 2)
MCP: none

# Encode a chunk to memory and decode from it

## What to build
- The encoder turns one or more 128-token chunks into memory vectors (K per chunk), using learned memory queries, the encoder LoRA, and a projection with RMSNorm.
- The decoder takes `[memory][control tokens][target]`, computes a teacher-forced loss on target tokens only, and can generate from memory.
- An `Engram` module wires these together. The model can overfit a single chunk to near-perfect reconstruction.

## Files to create/modify
- src/engram/model/encoder.py
- src/engram/model/decoder.py
- src/engram/model/memory.py
- src/engram/model/engram.py
- src/engram/model/base.py (named LoRA adapters: `encoder`, `decoder`)
- src/engram/tokens.py (adds `<full>`, `<terse>` control tokens)
- tests/test_encoder.py, tests/test_decoder.py, tests/test_overfit.py

## Test approach
- Shapes: N chunks → `[B, N*K, d]`.
- Trainable params are only the LoRA, memory queries, projection, and control-token embedding rows.
- Loss mask: memory and control positions contribute zero loss.
- Overfit test: about 200 steps on one chunk → greedy reconstruction matches ≥90% of tokens.

## Acceptance criteria
- [ ] The forward and backward pass gives a finite loss and gradients only on trainable params.
- [ ] Overfit test passes on CPU in under 5 minutes.
- [ ] `Engram.generate(text, density="full")` returns a string.

## Blocked by
01

## Gotchas
- Switch adapters per pass (`set_adapter("encoder")` → `set_adapter("decoder")`). Check gradients reach both adapters.
- Adding control tokens resizes embeddings. Train only the new rows (gradient mask or a separate embedding), keeping the base weights frozen.
- Encoder attention is causal: memory queries must come *after* the chunk so they can see all of it.
- Position ids for the decoder start at 0 for the memory vectors.
