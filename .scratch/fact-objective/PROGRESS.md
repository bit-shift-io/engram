# Progress — fact-objective

## Project mechanics
Test: .venv/bin/python -m pytest -q -m 'not slow'
Type-check: n/a
Lint: none configured

## Slice 01 — fact-weighting
- Built: fact-weighted `<full>` reconstruction loss behind `fact_weighting` / `fact_weight_cap`; weights from the decoder under zeroed memory in a no-grad pass.
- Files touched: src/engram/model/decoder.py, src/engram/train/trainer.py, src/engram/config.py, tests/test_decoder.py, tests/test_trainer.py
- New exports/interfaces: `decoder.fact_weights(token_losses [B,T], mask [B,T], cap) -> [B,T]`; `Decoder.token_losses(memory, control_ids, target_ids, memory_mask=None, target_mask=None, label_mask=None) -> [B,T]`; `Decoder.loss(..., weights=None)`; `trainer.fact_weighted_loss(eng, chunk_ids)`; `_batch_loss` reads `cfg.fact_weighting` ("full" batches only). Shared `Decoder._forward(..., use_labels)`, `_labels(...)`. Zeroing via `engram.eval.ablation.zero_memory`.
- Conventions confirmed: `tiny_engram` fixture; `cfg_for`/`build`/`data()`; `train()` returns run dir.
- Surprises: clip+rescale iterated until ≤cap and mean 1 (needs cap ≥ 1). Validation loss and other batch types unweighted. Step time +17–37% (2.95–3.85 s → 4.0–4.5 s, random tokens, noisy machine).
- Built by: slice-implementer · opus (inherited) · 35758 · verification passed first time

## Slice 02 — cloze-task
- Built: cloze task (`find_fact_spans`, example builder, `ClozeSampler`), trainer hook with `cloze_ratio`, and `cloze_val_loss_{real,shuffled,zeroed}` in validation.
- Files touched: src/engram/data/cloze.py (new), tests/test_cloze.py (new), src/engram/train/trainer.py, src/engram/config.py, src/engram/model/engram.py (off-list: `terse_loss(..., memory_transform=None)`)
- New exports/interfaces: `cloze.find_fact_spans(text)`; `make_cloze_example(tokenizer, chunk_ids[C], max_len) -> dict|None` ({chunks [1,C], question, target, density: "full"}); `ClozeSampler(sampler, ratio, generator).next()` → ("cloze", chunk); `trainer.cloze_examples(eng, chunk_ids, fallback=True)`; `_batch_loss` handles "cloze"; Config `cloze_ratio: float = 0.0`; metrics `n_cloze`, `cloze_val_loss_{real,shuffled,zeroed}`.
- Conventions confirmed: cloze uses `<answer>`-style prompt/target layout and existing `<full>` control id; test_cloze builds its own 64-word tokenizer.
- Surprises: cloze step cost on real K=16 not measured. Checked here on real WikiText val: 64/64 chunks have spans, 23% char coverage; prompts blank names correctly.
- Built by: slice-implementer · opus (inherited) · 42993 · verification passed first time

## Slice 03 — overnight-pipeline
- Built: fact-objective pipeline (K=16 train → recall + memory-use eval → stop if pass, else K=32 → summary), resume, configs, report helpers, docs.
- Files touched: configs/fact_k16.yaml, configs/fact_k32.yaml, scripts/fact_objective.py, src/engram/eval/report.py, tests/test_fact_pipeline.py, docs/training.md
- New exports/interfaces: `report.fact_arm(k, run, ev)`, `fact_summary_markdown(arms)`, `FACT_RECALL_PASS=0.15`, `FACT_BASELINES`, `FACT_STAGE2_RECALL`; `run_pipeline(pdir, runners, force_fail=False) -> "passed_k16"|"done"`; eval output `<run>/eval/fact_eval.json`.
- Conventions confirmed: importlib-loaded script + stubbed runners; state.json; imports `_derive`, `_latest_ckpt`, `_train_stage` from scripts/stage3b.py.
- Surprises: per-step 4.23 s (K=16), 4.60 s (K=32); validation ~5 min/pass → both arms ~11.3 h. Orchestrator raised val_interval 300→500 in both configs (monitoring only; steps unchanged) → ~10.5 h. K=16 Stage 1 recall baseline not measured ("n/a"). `--force-fail` flag added. Script always exits 0.
- Built by: slice-implementer · opus (inherited) · 53110 · verification passed first time
