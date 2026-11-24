# Architecture

```
                         ┌─────────────── Memory ───────────────┐
tokens in ──► Encoder ──►│ T0 working   fast, detailed, rolling │
                         │ T1 ideas     slot per idea (key/val) │◄──► Reasoner (latent steps)
                         │ T2 long-term persistent vector store │
                         └──────────────────────────────────────┘
                                          │
          control (density, query, slot id) ──► Decoder ──► tokens out
```

## Base LLM

A single pretrained decoder-only LLM serves as **both** encoder and decoder:

- **Frozen base weights.** These keep the language ability and keep training cheap.
- **Separate LoRA adapters** for the encoder role and the decoder role.
- A small (~0.5B–1.5B) open model to start. The model is a config setting, so it can be swapped.

Sharing one base model means memory vectors live in (roughly) the model's own embedding space. That makes the decoder's job easier.

## Encoder: tokens → memory

- The input is split into chunks of N tokens (e.g. 512).
- **K learned memory query tokens** are appended to each chunk: trainable embeddings `[M_1..M_K]`.
- The chunk and queries pass through the base model with the encoder LoRA. The final hidden states at the query positions become the chunk's **memory vectors** (K × d_model).
- This is the ICAE / gist-token pattern. The model can attend over the whole chunk and decides what to pack into the K slots.

## Memory tiers

| Tier | Holds | Size | Update rule |
|---|---|---|---|
| **T0 working** | Recent detail | K vectors per chunk, last W chunks | Rolling: new chunks are pushed, old ones fall off or are folded into T1 |
| **T1 ideas** | Discrete ideas | S slots, each a few vectors + key | Slot attention: write to the best-matching slot, or allocate a new one if nothing is close enough |
| **T2 long-term** | Persistent ideas | Unbounded, external | Promote a T1 slot when it's used or refined often enough, or on an explicit save; retrieve by key similarity |

T0 is built in Stage 1. T1 and T2 come later (see [training.md](training.md)).

## Decoder: memory → tokens

- Input: `[memory vectors] [control tokens] [optional prompt/question]`. Generation stops at EOS.
- **The decoder never sees input tokens.** That is what decouples output length from input length.
- Control tokens:
  - density: `<terse>`, `<normal>`, `<full>`, or a numeric token budget
  - task: `<reconstruct>`, `<summarise>`, `<answer>`
  - addressing (later): `<slot:i>` restricts or biases attention to one idea slot

## Reasoner (later stage)

- The same base model with a third LoRA, run for a few **latent steps** (Coconut-style). It reads slots, feeds the hidden state back as input, and writes updated slot values.
- Trained on tasks where the answer needs information from two or more slots combined.

## Data flow example

1. Read a 4,000-token article: 8 chunks × K=16 → 128 memory vectors in T0.
2. Fold T0 into T1: say 6 idea slots form.
3. "Summarise in 2 sentences" → decoder with `<terse>` over T1.
4. "What's the weakest argument in idea 4?" → decoder with `<slot:4> <answer>`.
5. The user saves idea 4 → it's promoted to T2 and can be retrieved in a later session.
