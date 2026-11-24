# Engram

Engram is a research prototype for an LLM that reads text into a **learned, tiered memory of ideas**, and then writes its output **from that memory** rather than from the raw input.

```
tokens in ──► Encoder (LLM) ──► Memory (T0 working / T1 ideas / T2 long-term) ──► Decoder (LLM) ──► tokens out
```

Goals:

- **Compression and long context.** Long inputs are folded into a small, fixed-size memory.
- **Output length decoupled from input length.** Verbose input can come out as precise, succinct output, at whatever density is asked for.
- **Ideas as addressable objects.** The model forms discrete "idea" slots. You can ask about a specific idea, refine it, or render it.
- **Reasoning in abstractions.** The model can think in latent space over ideas, not only in tokens.
- **Persistence.** Important ideas outlive a single context or session.

We start from an existing pretrained LLM for both encoding and decoding. Engram adds the memory in between and trains the minimum needed to make it work.

## Docs

| Doc | Contents |
|---|---|
| [docs/vision.md](docs/vision.md) | Motivation, goals, non-goals |
| [docs/architecture.md](docs/architecture.md) | Components, memory tiers, data flow |
| [docs/training.md](docs/training.md) | Objectives, staged curriculum, evals |
| [docs/related-work.md](docs/related-work.md) | Prior work we build on |
| [docs/implementation-plan.md](docs/implementation-plan.md) | Initial implementation plan (stages 1–2) |
| [docs/open-questions.md](docs/open-questions.md) | Undecided design choices |
