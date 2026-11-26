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
- **Answer task** — `<answer>` control: memory + question → short answer. Stage 3.
- **Text-QA control** — baseline reading the paragraph as plain text instead of memory, trained the same way. The denominator for the memory QA ratio.
- **`<full>` gap** — mean `<full>` val loss with zeroed memory minus with real memory (nats). Stage 2 ≈ 1.0.
- **Regression gate** — a Stage 3 metric (`<full>` gap, terse/full length ratios, `<terse>` ROUGE-L) checked against the Stage 2 band; outside it = regression. Written to `<run>/eval/stage3.md`.
- **Answer recall** — share of held-out SQuAD paragraphs whose `<full>` reconstruction from memory contains a gold answer. Measures whether memory keeps specifics. K=16 Stage 2: 7.5%.
- **Recall gate** — answer recall threshold (25%) after the K=32 warm-up; below it the joint stage does not run.

## Fact token
- **Definition:** A chunk token inside a detected name, number or date span.
- **Boundary:** Detected by text patterns, not NER. Not the same as an answer span.

## Fact weighting
- **Definition:** Per-token reconstruction loss weight from the frozen decoder's loss on that token with zeroed memory.
- **Boundary:** Weights are detached and normalised to mean 1 per chunk; they never carry gradient.

## Cloze task
- **Definition:** The decoder sees a chunk with fact spans blanked and outputs the fact spans, using memory.
- **Boundary:** A training task only. Answer recall, not cloze loss, measures fact retention.
