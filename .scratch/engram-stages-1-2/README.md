# Engram Stages 1–2: memory autoencoder with density control

## Problem
- No evidence yet that a pretrained LLM can write text into a small **memory** and rebuild it from that memory alone.
- No evidence yet that output length can be set independently of input length.
- Only hardware: an Intel MacBook (i7-9750H, 16GB RAM), CPU only.

## Solution
- A frozen small base LLM encodes text chunks into K **memory vectors** each, using an encoder LoRA and learned memory queries.
- The same base LLM, with a decoder LoRA, generates text from memory and control tokens only. It never sees the input tokens.
- Two **densities**: `<full>` rebuilds the input; `<terse>` writes a short summary.

## Out of Scope
- T1 idea slots, `<slot:i>` addressing, latent reasoner, T2 persistence.
- QA objective, paraphrase consistency, capacity penalties.
- `<normal>` density and numeric length budgets.
- LLM-generated training data (Claude API).
- GPU training runs and W&B logging.

## Cross-cutting Constraints
- Must install and run on Intel macOS: `torch==2.2.2`, Python 3.12, managed with `uv`.
- Device is a single config field (`cpu` default, `cuda` allowed). No hard-coded device anywhere.
- Base model weights stay frozen. Only LoRA adapters, memory queries and new control-token embeddings train.
- Each training run must finish in under ~4 hours on the CPU, using the default config.
- Every run writes a config snapshot and metrics to its own run directory.

## Acceptance Criteria
- [ ] `uv sync` then `uv run pytest` passes on the Intel Mac.
- [ ] Stage 1 run (K=16, 128-token chunks, held-out chunks, teacher-forced): real-memory loss ≥ 0.5 nats below shuffled memory, and token accuracy ≥ 10 points above shuffled. (Relaxed 2026-11-25 from greedy reconstruction ≥ 80% — see DECISIONS.md.)
- [ ] Ablation: zeroed and shuffled memory are both clearly worse than real memory.
- [ ] Stage 2 run: for the same memory, `<terse>` output is ≤ 25% of the `<full>` output length on average.
- [ ] After Stage 2, the Stage 1 real-vs-shuffled `<full>` loss gap stays ≥ 0.4 nats.
- [ ] `<terse>` ROUGE-L on held-out XSum is reported next to the base model's direct-summary baseline.
- [ ] `scripts/demo.py "<text>"` prints the compression ratio, plus `<full>` and `<terse>` renders.

## References
- [docs/architecture.md](../../docs/architecture.md), [docs/training.md](../../docs/training.md), [docs/related-work.md](../../docs/related-work.md)
- [docs/CONTEXT.md](../../docs/CONTEXT.md) glossary
- ICAE (Ge et al., 2023): closest template for this work.
