# Repo memory

- [intel-mac-torch-pin](intel-mac-torch-pin.md) — Intel Mac, CPU only: torch 2.2.2, Python 3.12, numpy<2, pyarrow<17
- [hf-weights-and-sandbox](hf-weights-and-sandbox.md) — HF_HOME=.hf in repo, offline tests, downloads need sandbox off (certifi .pem blocked)
- [peft-multi-adapter-grads](peft-multi-adapter-grads.md) — use use_adapter(), not set_adapter (freezes the other LoRA); clip grads at 1.0
