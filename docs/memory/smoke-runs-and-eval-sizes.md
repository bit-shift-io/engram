---
name: smoke-runs-and-eval-sizes
description: --smoke may only shrink steps and eval sizes to already-cached n (QA n=40 or 200); smoke output stays out of git; in-training QA checks are too small to judge QA progress
metadata:
  type: context
---

**Why:** Data caches are keyed by sample count and HF runs offline, so an uncached eval size breaks or rebuilds data. A pipeline smoke run with the QA eval at n=200 took ~1 h; at n=40 (also cached) ~17 min, almost all training. Old smoke checkpoints (191 MB) were once committed by accident.
**How to apply:** When adding `--smoke` to a script, or judging a run from its metrics file.

- Smoke configs shrink `steps`, val intervals and eval sizes, never `train_samples` / `qa_*_samples` (see `SMOKE` and `_derive` in `scripts/squad_cloze.py`, `scripts/fact_qa.py`). Cached QA sizes: 40 and 200.
- Smoke output goes to `runs_smoke_*/` at the repo root; `.gitignore` excludes it with `runs/`. Never commit checkpoints.
- In-training QA checks (`qa_real_f1` in metrics.jsonl) use a handful of questions: in the 2026-11-28 Stage B they read 0.038 with real = shuffled for all 500 steps, while the n=200 eval gave 0.176 vs 0.068. Judge QA only from a full-size eval.
- Smoke Stage A is too short to pass the fact gate, so `fact_qa.py --smoke` ignores the gate; the gate-fail path is covered only by stub tests. See [[fact-qa-stages-result]].
