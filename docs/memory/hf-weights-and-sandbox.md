---
name: hf-weights-and-sandbox
description: Model weights and datasets cache in the repo's .hf/ (gitignored); tests run offline; downloads need the Claude sandbox off
metadata:
  type: convention
---

**Why:** Caches in `$TMPDIR` or `~/.cache` vanish or are unreadable inside the Claude Code sandbox, and the sandbox blocks reading `*.pem` (including certifi's CA bundle), so HuggingFace downloads fail with "Could not find a suitable TLS CA certificate bundle".
**How to apply:** Whenever you run anything that loads SmolLM2, WikiText or XSum — tests, `scripts/train.py`, `scripts/eval.py`, `scripts/demo.py`.

- Set `HF_HOME=<repo>/.hf`. `.hf/`, `data/cache/` and `runs/` are gitignored.
- Tests and scripts that only need cached data: also set `HF_HUB_OFFLINE=1`.
- A first download of a new model or dataset (or split) needs network with `HF_HUB_OFFLINE` unset, run outside the sandbox. After that it works offline.
