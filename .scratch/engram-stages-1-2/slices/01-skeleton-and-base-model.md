Status: done
Complexity: 5 (design 1, scope 2, tests 0, hazards 2)
MCP: none

# Project skeleton loads base model and generates

## What to build
- A `uv` project, with `torch==2.2.2` and Python 3.12, that installs on Intel macOS.
- Config loads from YAML into dataclasses (model name, device, chunk size, K, seed, run dir).
- `scripts/generate.py "prompt"` loads the base model from config and prints a completion.

## Files to create/modify
- pyproject.toml, .python-version, .gitignore
- configs/default.yaml
- src/engram/__init__.py
- src/engram/config.py
- src/engram/model/base.py
- scripts/generate.py
- tests/test_config.py, tests/test_base_model.py

## Test approach
- Config: load YAML, override a field, reject an unknown field.
- Base model: loads on CPU, all params frozen, generates ≥1 new token. Mark it `slow`; use a tiny random-init model for the fast path if available.

## Acceptance criteria
- [ ] `uv sync && uv run pytest` passes on the Intel Mac.
- [ ] `uv run scripts/generate.py "The capital of France is"` prints a completion.
- [ ] Changing `device` in the config is the only change needed to target CUDA.

## Blocked by
None — can start immediately.

## Gotchas
- `torch==2.2.2` wheels exist only for Python ≤3.12 on x86_64 macOS. Pin it; the system Python is 3.15.
- Pick `transformers` and `peft` versions that still support torch 2.2. Recent `transformers` releases require a newer torch, so check before upgrading.
- torch 2.2 wheels predate NumPy 2. Pin `numpy<2`.
- Install `uv` first if missing (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`).
