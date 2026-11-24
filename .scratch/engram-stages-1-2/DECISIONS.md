# Decisions

### Q1: Compute
**Decision:** Local Intel Mac only, CPU training.
- **Why:** The user's only hardware for now; no cloud.
- **Implication:** Tiny model, short chunks, small K, runs of a few hours.
- **Alternatives considered:** Colab/Kaggle GPUs. Declined for now; config keeps a `device` field so we can switch later.

### Q2: PyTorch version
**Decision:** Pin `torch==2.2.2`, Python 3.12, `uv`-managed.
- **Why:** 2.2.2 is the last PyTorch release with Intel macOS wheels. System Python 3.15 is too new.
- **Implication:** No newer torch features. `transformers`/`peft` versions must be ones that still support torch 2.2.
- **Alternatives considered:** Build torch from source. Rejected: slow and fragile.

### Q3: Base model
**Decision:** SmolLM2-135M (base, not instruct) as the default; configurable.
- **Why:** Small enough for CPU LoRA training on 16GB; modern tokenizer; Apache-2.0.
- **Implication:** Hidden size 576. Reconstruction quality caps lower than in ICAE.
- **Alternatives considered:** Qwen 0.5B. Rejected: too slow on CPU.

### Q4: Chunk size and K
**Decision:** 128-token chunks; K=16 default; sweep K ∈ {4, 8, 16, 32}.
- **Why:** 8× compression at K=16 gives a meaningful test while staying cheap.
- **Alternatives considered:** 512-token chunks. Rejected: too slow on CPU.

### Q5: Densities
**Decision:** Two only: `<full>` and `<terse>`.
- **Why:** Two levels prove output length is decoupled from input length; existing data covers both.
- **Alternatives considered:** Add `<normal>`. Deferred until generated data exists.

### Q6: Data
**Decision:** Existing datasets only. WikiText-103 for reconstruction; XSum for `<terse>`.
- **Why:** Free and offline once cached. XSum summaries are one sentence, which gives a strong length contrast.
- **Alternatives considered:** Claude-generated rewrites. Deferred.

### Q7: Multi-chunk memory for summaries
**Decision:** Long inputs are encoded chunk by chunk; the memory vectors from all chunks are concatenated (T0) and the decoder reads them all. Cap at 4 chunks (512 tokens).
- **Why:** An XSum summary covers the whole article, not just its first 128 tokens.
- **Implication:** XSum articles are truncated to 512 tokens. Each chunk is encoded independently (no carry-over between chunks yet).
- **Alternatives considered:** Carry memory between chunks (RMT-style). Deferred; adds sequential cost.

### Q8: Memory injection
**Decision:** Final hidden states at the memory-query positions, passed through a learned linear projection + RMSNorm, then fed to the decoder through `inputs_embeds`.
- **Why:** Final hidden states differ in scale from input embeddings; the projection is cheap insurance.

## Assumptions
- About 5–20k WikiText chunks are enough for the reconstruction signal at this scale.
- The HF datasets cache works offline after the first download.

## Trade-offs
- Speed over quality: a tiny model and short chunks make results look worse than in the paper, but iteration takes hours, not days.

## New CONTEXT.md terms
- Memory, memory vector, memory query, chunk, density, control token, compression ratio, memory ablation.

## 2026-11-25 — Relaxed Stage 1/2 targets after CPU training runs
- Decision: Stage 1 passes on a teacher-forced real-vs-shuffled gap on held-out chunks (loss ≥ 0.5 nats, token accuracy ≥ +10 pts), not ≥ 80% greedy reconstruction. Stage 2 keeps the ≤ 25% length target and must keep the `<full>` gap ≥ 0.4 nats.
- Why: on the Intel CPU, a frozen SmolLM2-135M decoder learns to read soft-prompt memory very slowly. Full-config runs ignored memory entirely; the best run (512 chunks × ~17 epochs, ~2.3 h) reached a 0.28-nat held-out gap (acc 0.471 vs 0.436). 80% greedy reconstruction needs GPU-scale compute. The relaxed targets still prove text goes into memory and is read back out.
- Also decided: decoder LoRA off for Stage 1 (it learns a shortcut and ignores memory). Next: a decoder adapter that acts only on memory positions, compared against the plain frozen decoder before a long overnight run.
