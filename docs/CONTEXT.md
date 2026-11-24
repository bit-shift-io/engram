# Glossary

- **Chunk** — fixed-length token window (default 128) encoded as a unit. *Not* a semantic unit.
- **Memory** — all memory vectors the decoder reads for one input. Stages 1–2 have only T0.
- **Memory vector** — one `d_model` vector produced by the encoder. K per chunk.
- **Memory query** — learned embedding appended to a chunk; its final hidden state becomes a memory vector.
- **Tier (T0/T1/T2)** — working memory / idea slots / long-term store. See architecture.md.
- **Idea slot** — a T1 unit holding one idea; addressable with `<slot:i>`. Not built yet.
- **Density** — how much detail the decoder renders: `<full>` (reconstruct) or `<terse>` (summary). Not a token count.
- **Control token** — special token telling the decoder density or task.
- **Compression ratio** — input tokens ÷ memory vectors.
- **Memory ablation** — replacing memory with zeros or another input's memory to test whether the decoder uses it.
