# Repo memory

- [intel-mac-torch-pin](intel-mac-torch-pin.md) — Intel Mac, CPU only: torch 2.2.2, Python 3.12, numpy<2, pyarrow<17
- [hf-weights-and-sandbox](hf-weights-and-sandbox.md) — HF_HOME=.hf in repo, offline tests, downloads need sandbox off (certifi .pem blocked)
- [peft-multi-adapter-grads](peft-multi-adapter-grads.md) — use use_adapter(), not set_adapter (freezes the other LoRA); clip grads at 1.0
- [decoder-ignores-memory](decoder-ignores-memory.md) — decoder LoRA shortcut ignores memory: train frozen + memory-only k/v adapter; LoRA later at low LR
- [prefix-control-token-ignored](prefix-control-token-ignored.md) — one <full>/<terse> token gets overridden; use control_vectors, a balanced mix, and replay old modes on each new mode's input type (Stage 3 <full> collapsed on SQuAD)
- [memory-necessity-eval](memory-necessity-eval.md) — judge memory by real vs shuffled vs zeroed on held-out chunks, not by loss
- [reconstruction-mix-dilutes-wikitext](reconstruction-mix-dilutes-wikitext.md) — adding SQuAD chunks by replacing WikiText (plus EOS padding) stalled K=32 warm-up; add on top, strip padding; K=32 plateaus ~1200 steps
- [reconstruction-keeps-gist-not-facts](reconstruction-keeps-gist-not-facts.md) — token-averaged reconstruction keeps gist (F1 ~0.2) but not facts (answer recall 5–7.5%, K=16 and K=32); needs a fact-rewarding objective; WikiText cloze alone did not carry over to SQuAD; SQuAD gold-answer cloze did (recall 0.20)
- [k32-fails-with-extra-objectives](k32-fails-with-extra-objectives.md) — fresh K=32 stalls (real = shuffled) with SQuAD mix or fact objectives; only plain WikiText K=32 learns; prefer K=16
