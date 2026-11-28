# Progress — fact-qa-stages

## Project mechanics
Test: HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q -m "not slow"
Type-check: n/a
Lint: none configured

## Slice 01 — stage-a-config-and-load
- Built: Stage A density config, init from the SQuAD cloze checkpoint where only control-vector and decoder-LoRA keys may be missing, and a smoke run that saved a checkpoint.
- Files touched: configs/stage_a_density.yaml (new), src/engram/train/trainer.py, tests/test_trainer.py
- New exports/interfaces: `engram.train.trainer._may_start_fresh(name: str) -> bool` (private). `load_checkpoint(eng, path, opt=None) -> int` now raises `KeyError` when any other trainable parameter is missing (also on --resume). Init checkpoint file: runs/squad_cloze/train/20261127-163240/ckpt-2000.pt. Run: `.venv/bin/python scripts/train.py --config configs/stage_a_density.yaml --init-from <ckpt>`.
- Conventions confirmed: tests use `cfg_for`/`build`/`data` from tests/test_trainer.py with tiny conftest models. Smoke runs shrink only steps and val sizes, never sample counts (cache keys), like `SMOKE`/`_derive` in scripts/squad_cloze.py. Cloze replay only replaces 1-chunk `<full>` batches.
- Surprises: scripts/train.py has no `--smoke` flag (pipeline slice should copy SMOKE/_derive from scripts/squad_cloze.py). `fact_weighting: true` set to match init checkpoint. Sample counts copy the SQuAD cloze run (train 2000, qa_train 4000, qa_val 40). 500 steps ≈ 3.3 h CPU.
- Built by: slice-implementer · inherited model · 50121 · verification passed first time

## Slice 02 — fact-gate
- Built: `fact_gate` returns `facts_passed` (cloze recall real ≥ 0.13 and zeroed−real answer_loss ≥ 1.5, boundaries pass) separately from `length_warnings` (from `stage3_gates(row, None)["regressions"]`, filtered to terse_article_ratio / full_article_ratio).
- Files touched: src/engram/eval/report.py, tests/test_trainer.py
- New exports/interfaces: `fact_gate(cloze, density, val) -> {facts_passed, checks: {cloze_recall|cloze_gap: {value, threshold, passed}}, length_warnings, length_values}`; `fact_gate_markdown(g) -> str`; constants `FACT_GATE_RECALL`, `FACT_GATE_GAP`, `LENGTH_BANDS`. `cloze` is the `evaluate_cloze_recall` shape: `cloze["real"]["cloze_recall"]`, `cloze[...]["answer_loss"]` for real/zeroed.
- Conventions confirmed: gate constants and markdown helpers live in report.py; gate tests live in tests/test_trainer.py beside stage3_gates tests (no test_report.py).
- Surprises: a ratio missing from `density` counts as a length warning.
- Built by: slice-implementer · sonnet/low · 22633 · verification passed first time

## Slice 03 — stage-b-config-and-final
- Built: configs/stage_b_answer.yaml (stage3.yaml + Stage A sample counts + cloze replay), and `fact_qa_final` / `fact_qa_final_markdown` in report.py.
- Files touched: configs/stage_b_answer.yaml (new), src/engram/eval/report.py, tests/test_report.py (new)
- New exports/interfaces: `fact_qa_final(qa, cloze_recall, density, val, control) -> {qa, qa_n, control_f1, ratio, ratio_passed, warnings, metrics, cloze_recall}`; `qa` = `{n, real, shuffled, zeroed}` each `{em, f1}`; `cloze_recall` = `{"cloze_recall": float}`; `ratio_passed` True/False vs `K32_THRESHOLD` (0.5), None if control missing. `fact_qa_final_markdown(s) -> str`.
- Conventions confirmed: report gates reuse `stage3_gates`, `STAGE2_GATES`, `LENGTH_BANDS`; `warnings` covers only STAGE2_GATES metrics.
- Surprises: cloze replay ratios (cloze_ratio 0.3, squad_cloze_ratio 0.3) are a guess; ~12% of batches become cloze. steps 500, ~2.2 h estimated. Config `qa_val_samples` is 40 — the pipeline must run the final QA eval at n=200 explicitly to match runs/text_qa_control.json.
- Built by: slice-implementer · sonnet · 26563 · verification passed first time

## Slice 04 — pipeline
- Built: scripts/fact_qa.py — resumable Stage A → evals → fact gate → Stage B → evals → summary.{json,md}; stub-runner tests for gate pass/fail, length miss, resume, smoke gate bypass.
- Files touched: scripts/fact_qa.py, tests/test_fact_qa_pipeline.py (smoke output: runs_smoke_fact_qa/, runs_smoke_fact_qa_n40/)
- New exports/interfaces: `run_pipeline(pdir, runners, ignore_gate=False) -> "done"|"gate_failed"`; `make_runners(smoke, init_from=INIT_CKPT)` → stage_a/eval_a/stage_b/eval_b; CLI `--dir --smoke --init-from`. Real run: `HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 .venv/bin/python scripts/fact_qa.py` → runs/fact_qa, exit 0 only when `done`. State in state.json; final QA at QA_N=200 (smoke 40); summary records n.
- Conventions confirmed: reuses `_derive`, `_latest_ckpt`, `_train_stage` from scripts/stage3b.py; eval outputs cloze_recall.json, qa.json, density.json.
- Surprises: smoke Stage A fails the fact gate (gap 1.44 < 1.5), so --smoke ignores the gate (recorded in summary); real gate_failed path covered only by stub test. Smoke ~17 min wall (training dominates), not "a few minutes". Resume confirmed: re-running on a finished dir returns instantly.
- Built by: slice-implementer · sonnet · 69450 · needed 1 re-run (smoke QA n)

## Slice 05 — run-and-record
- Built: real run of scripts/fact_qa.py (runs/fact_qa/, ~7 h); result in docs/memory/fact-qa-stages-result.md + README index.
- Files touched: docs/memory/fact-qa-stages-result.md (new), docs/memory/README.md
- New exports/interfaces: none
- Conventions confirmed: memory entries use name/description/metadata frontmatter plus Why / How to apply.
- Surprises: gate passed (recall 0.133, gap 2.05) but `<full>` length 0.148 (band 0.8–1.05); QA ratio 0.256 FAIL; memory QA F1 0.176 below base zero-shot 0.196.
- Built by: inline · opus · n/a · verification passed first time
