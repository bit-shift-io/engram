# Open questions

- **Scaling up.** When to move beyond SmolLM2-135M on CPU, and to what hardware.
- **Memory vectors: hidden states vs. projected.** Is the last-layer hidden state used directly, or passed through a learned projection back into input-embedding space?
- **Chunk size and K.** Fixed per run, or should compression adapt to how dense the content is?
- **Variable-size concepts.** Should a concept be allowed to span a sentence *or* a paragraph? Possible approaches: learned boundaries (H-Net style) or slot sparsity.
- **T1 slot formation.** Slot attention over T0, or a dedicated encoder pass?
- **Slot addressing.** Hard masking vs. soft attention bias for `<slot:i>`.
- **T2 store.** Plain vector DB (FAISS) vs. learned retrieval; how promotion is decided.
- **Synthetic data.** Which LLM generates the QA and rewrites, and how do we avoid leaking its style into the outputs?
