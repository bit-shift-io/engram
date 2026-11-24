---
name: intel-mac-torch-pin
description: Dev machine is an Intel Mac — torch pinned to 2.2.2, Python 3.12, CPU only
---

The only dev and training machine is an Intel MacBook (i7-9750H, 6 cores, 16GB RAM, no MPS).

- `torch==2.2.2` is the last release with Intel macOS wheels. Don't bump torch.
- Python 3.12, managed by `uv`. The system Python (3.15) isn't supported.
- `numpy<2`. Pin `transformers`/`peft` to versions compatible with torch 2.2.
- Use `torch.set_num_threads(6)`.
- Keep the device in config. Code must also run on `cuda` without edits.
