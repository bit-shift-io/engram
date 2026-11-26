# Repo memory

- [intel-mac-torch-pin](intel-mac-torch-pin.md) — Intel Mac, CPU only: torch 2.2.2, Python 3.12, numpy<2, pyarrow<17
- [hf-weights-and-sandbox](hf-weights-and-sandbox.md) — HF_HOME=.hf in repo, offline tests, downloads need sandbox off (certifi .pem blocked)
- [peft-multi-adapter-grads](peft-multi-adapter-grads.md) — use use_adapter(), not set_adapter (freezes the other LoRA); clip grads at 1.0
- [decoder-ignores-memory](decoder-ignores-memory.md) — decoder LoRA shortcut ignores memory: train frozen + memory-only k/v adapter; LoRA later at low LR
- [prefix-control-token-ignored](prefix-control-token-ignored.md) — one <full>/<terse> token gets overridden; use control_vectors, a balanced mix, and replay old modes on each new mode's input type (Stage 3 <full> collapsed on SQuAD)
- [memory-necessity-eval](memory-necessity-eval.md) — judge memory by real vs shuffled vs zeroed on held-out chunks, not by loss
