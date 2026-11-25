---
name: memory-necessity-eval
description: Judge memory by teacher-forced loss with real vs shuffled vs zeroed memory on held-out chunks, never by training loss alone
metadata:
  type: convention
---

**Why:** Training and validation loss can fall nicely while the decoder ignores memory, because it can write plausible text from the base model's language knowledge. Several runs looked fine on loss but had no real-vs-shuffled gap.
**How to apply:** When reading any training run's results, or picking between configs.

- Compare three conditions on the same held-out chunks: real memory, shuffled (another chunk's memory, `memory.roll(1, 0)`) and zeroed. Shuffled is the key one: zeroed is out of distribution, so it always looks bad.
- Use teacher-forced loss and token accuracy; greedy generation is slow on CPU and noisy.
- Any custom eval must open the same decoder context as training (memory-adapter window, control vectors), or it silently reports "no gap". Use `Decoder.loss` or `Decoder.generate` rather than calling the LM directly.
- Expect loss gaps to grow well before accuracy gaps: memory carries gist (it makes the right words likelier) before it carries exact text.
