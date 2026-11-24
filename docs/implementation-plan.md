# Initial implementation plan (Stages 1–2)

**Goal:** show that a frozen pretrained LLM can encode text chunks into K memory vectors, rebuild the text from memory alone (Stage 1), and render the same memory at two densities, `<full>` and `<terse>` (Stage 2).

## Constraints
- Runs on an Intel MacBook (i7-9750H, 16GB RAM), CPU only. See [memory/intel-mac-torch-pin.md](memory/intel-mac-torch-pin.md).
- `torch==2.2.2`, Python 3.12, `uv`, `transformers` + `peft`.
- Device is a config field, so the same code runs on CUDA later.

## Defaults
| Setting | Value |
|---|---|
| Base model | SmolLM2-135M (frozen) |
| Chunk size | 128 tokens |
| K (memory vectors per chunk) | 16 (sweep 4–32) |
| Max chunks per input | 4 (512 tokens) |
| Densities | `<full>`, `<terse>` |
| Reconstruction data | WikiText-103 |
| Terse data | XSum |

## Steps
1. Project skeleton; base model loads and generates.
2. Encode → memory → decode round trip; overfit a single chunk.
3. Stage 1 reconstruction training on WikiText.
4. Reconstruction and memory-ablation eval.
5. Stage 2 density control with XSum.
6. Density eval and `demo.py`.

## Repo layout
```
configs/            default, smoke, stage1, stage2 YAML
src/engram/
  config.py tokens.py
  model/            base, encoder, decoder, memory, engram
  data/             chunking, reconstruction, compression, mixed
  train/            losses, trainer
  eval/             reconstruct, ablation, density, report
scripts/            generate, train, eval, demo
tests/
```

## Risks
| Risk | Mitigation |
|---|---|
| Decoder ignores memory | Ablation eval (real vs zeroed vs shuffled memory) |
| Memory vectors out of distribution for the decoder | Same base model for both roles; projection + RMSNorm |
| `<terse>` crowds out `<full>` | ≥50% `<full>` batches; track reconstruction accuracy in Stage 2 |
| CPU too slow | 135M model, 128-token chunks, runs capped at ~4 h |

## After this
Stage 3 QA → Stage 4 T1 idea slots → Stage 5 latent reasoner → Stage 6 T2 persistence.
