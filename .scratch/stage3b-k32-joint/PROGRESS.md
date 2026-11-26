# Progress — stage3b-k32-joint

## Project mechanics
Test: .venv/bin/python -m pytest -q -m 'not slow'
Type-check: n/a
Lint: none configured

## Slice 01 — answer-recall-eval
- Built: `eval.py --suite recall` runs the `<full>` reconstruction check and writes `<run>/eval/recall.{json,md}`.
- Files touched: src/engram/eval/recall.py (new), scripts/eval.py, tests/test_recall_eval.py (new). Baseline run also added runs/20261125-163535/eval/recall.{json,md} (eval output only; ckpts/config untouched).
- New exports/interfaces: engram.eval.recall: `answer_in_text(answers, text) -> bool`; `evaluate_recall(eng, examples) -> {k, n, answer_recall, recon_token_f1, length_ratio}`; `recall_markdown(results) -> str`; `write_recall_report(run_dir, result) -> Path`. CLI: `scripts/eval.py --run <run> --suite recall --n 40`.
- Conventions confirmed: tests use `tiny_engram` fixture + `make_qa_example`; real-weight runs need `HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1`; recall loads 200 dev examples and slices to --n (cache keyed by n).
- Surprises: K=16 Stage 2 baseline: answer recall 0.075, recon F1 0.261, length ratio 0.500 (n=40). Recall is real-memory only (spec asks no shuffled/zeroed for this metric).
- Built by: slice-implementer · sonnet · 26942 · verification passed first time

## Slice 02 — k32-warmup
- Built: K=32 warm-up config with WikiText/SQuAD-paragraph reconstruction mixing; `answer_recall` / `recon_token_f1` logged each validation step.
- Files touched: configs/stage3b_warmup.yaml, src/engram/config.py, src/engram/data/reconstruction.py, src/engram/train/trainer.py, scripts/train.py (off-list: wires mixing + answer_val loading), tests/test_chunking.py, tests/test_trainer.py. Smoke dir runs_smoke_k32/20261126-111527 left in repo root (deletion denied; user to remove).
- New exports/interfaces: Config `squad_recon_share: float = 0.0`, `val_recall_samples: int = 0`. `reconstruction.squad_paragraph_chunks(examples, n)`; `reconstruction.mix_chunks(wiki, squad_examples, share, seed=0)`. `trainer.train` adds recall metrics to val row when `val_recall_samples > 0` and `answer_val` given. `scripts/train.py` loads `answer_val` when `answer_ratio > 0` or `val_recall_samples > 0`.
- Conventions confirmed: tests use `cfg_for`, `build`, `data()` in test_trainer.py; real runs `HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1` (SQuAD train 4000-example cache now present in data/cache).
- Surprises: ~3.9 s/step at batch 4, K=32 → steps=2600 (~2.8 h + val; val recall generation not timed separately). squad_recon_share 0.5 uses 4000 train examples. SQuAD repeats paragraphs per question → dedupe by paragraph. Last chunk EOS-padded.
- Built by: slice-implementer · sonnet · 32643 · verification passed first time

## Slice 03 — joint-stage
- Built: K=32 joint stage config (40/35/25, half of `<full>` from SQuAD paragraphs), SQuAD `<full>` source in `MixedSampler`, SQuAD gates beside the XSum bands, and a matching text-QA config.
- Files touched: configs/stage3b_joint.yaml (new), configs/text_qa_3b.yaml (new), src/engram/data/mixed.py, src/engram/train/trainer.py, src/engram/eval/report.py, src/engram/config.py, tests/test_trainer.py; smoke output runs_smoke_k32/20261126-112051, runs_smoke_k32/20261126-112642.
- New exports/interfaces: `Config.squad_full_ratio: float = 0.0`; `MixedSampler(..., squad_full=None, squad_ratio=0.0)` yields `("squad_full", list)`; `report.SQUAD_GATES = {"squad_full_length_ratio": (0.50, 0.70, 1.30), "answer_recall": (0.075, 0.075, None)}` (only checked when the val row has the metric); val key `squad_full_length_ratio`; metrics key `n_squad_full`; `trainer._squad_paragraph_examples(examples, eos_id)`. Joint config `--init-from` is a placeholder until warm-up run exists. text_qa_3b: 240 steps, writes runs/text_qa_control_3b.json.
- Conventions confirmed: `load_config` rejects unknown keys; cached data matches only existing sample counts (val_samples 64, terse_val_samples 32, qa_val_samples 40, qa_train_samples 4000).
- Surprises: non-SQuAD half of `<full>` split equally XSum whole-article / 1-chunk WikiText (8.75% each; matches DECISIONS Q4). SQuAD gate bands unreviewed. ~13.2 s/step (6–31 s; XSum `<full>` slow), planned 15 → steps 600, val every 120 (~9 min each), ckpt every 60 → ~3.25 h.
- Built by: slice-implementer · opus (inherited) · 55503 · verification passed first time

## Slice 04 — pipeline
- Built: scripts/stage3b.py unattended pipeline with recall gate, disk-backed resume (state.json), gate + final summaries; launch command in docs/training.md.
- Files touched: scripts/stage3b.py (new), src/engram/eval/report.py, tests/test_pipeline.py (new), docs/training.md. Smoke output runs_smoke_k32/stage3b/.
- New exports/interfaces: report.RECALL_GATE=0.25, recall_gate, recall_gate_markdown, stage3b_final, stage3b_final_markdown; stage3b.run_pipeline(pdir, runners, threshold) -> "done"|"gate_failed"; CLI `scripts/stage3b.py [--dir D] [--smoke] [--gate-threshold T]` (default dir runs/stage3b).
- Conventions confirmed: tests load scripts via importlib; pipeline calls train/eval/train_text_qa main() in-process.
- Surprises: total estimate ~7.5–7.8 h (borderline; cut options: final qa 100 examples, sparser warm-up val recall). Offline cache keyed by sample count → smoke keeps cached eval sizes. Smoke needs --gate-threshold 0. Gate re-applied per run; failed re-run leaves earlier final.*. Final markdown reuses stage3_markdown with stale "recommend a K=32 arm" wording.
- Built by: slice-implementer · opus (inherited) · 52679 · verification passed first time
