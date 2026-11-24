# Vision

## The idea

Today's LLMs map tokens to tokens. Everything they "know" about the input lives in a KV cache that grows with the input, and the length and style of the output follow the input closely.

Engram puts a **bottleneck memory** between reading and writing:

1. **Read.** An LLM encodes the input stream and *writes* into memory.
2. **Remember.** The memory holds distilled "concepts". The model decides their shape through training; we don't hand-design them.
3. **Write.** An LLM decodes text *from memory*, at whatever length and density is requested.

This draws on Next-Concept Modelling (predicting in a concept space instead of token space), but here the concept space is **learned end to end** instead of fixed, as it is with pretrained sentence embeddings.

## Goals

1. **Compression and long context.** Memory size is fixed or grows slowly, regardless of input length.
2. **Length decoupling.** The decoder sees memory, not the input. One memory can be rendered terse or in full.
3. **Addressable ideas.** Memory is organised into idea slots that can be:
   - formed: created, or merged into an existing slot
   - queried: "what is the weakness of idea #3?"
   - refined: updated through latent reasoning
   - rendered: turned into text at a chosen density
4. **Latent reasoning.** Multi-step thinking over idea slots without producing tokens.
5. **Persistence.** Ideas that are used or refined often get promoted to a long-term store and can be retrieved in later sessions.

## Non-goals (for now)

- Training a base LLM from scratch. We reuse a pretrained one.
- Matching frontier-model quality. The aim is to show that the memory *works and is used*.
- Multimodal input.

## How we'll know it works

- **Memory is necessary.** Ablating memory significantly degrades output.
- **Memory is sufficient.** Output from memory alone answers questions about the input nearly as well as output from the full input.
- **Ideas are localised.** Ablating one slot degrades answers about *that* idea and leaves the others mostly unaffected.
- **Density is controllable.** The same memory rendered `<terse>` stays well under the input length while keeping its QA coverage.
