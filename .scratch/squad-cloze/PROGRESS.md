# Progress — squad-cloze

## Project mechanics
Test: .venv/bin/python -m pytest -q -m 'not slow'
Type-check: n/a
Lint: none configured

## Slice 01 — squad-cloze-training
- Built: gold-answer SQuAD cloze examples (first chunk per distinct paragraph), `squad_cloze_ratio` sampler share, `squad_cloze_val_loss_{real,shuffled,zeroed}` val logging.
- Files touched: src/engram/data/cloze.py, src/engram/train/trainer.py, src/engram/config.py, tests/test_cloze.py, tests/test_trainer.py
- New exports/interfaces: `Config.squad_cloze_ratio: float = 0.0`; `squad_cloze_examples(tokenizer, examples, max_len, exclude=None) -> list[dict]` (takes `load_qa_examples` output, one example per paragraph); `make_gold_cloze_example(tokenizer, chunk_ids, answers, max_len)`; `paragraph_answers(examples)`; shared `_layout` helper; `ClozeSampler(sampler, ratio, gen, squad_cloze=None, squad_ratio=0.0, batch_size=1)` may return `("squad_cloze", list[dict])`; train log gains `n_squad_cloze`.
- Conventions confirmed: trainer tests use `cfg_for`/`build`/`data()`; cloze tests use `fact_tokenizer`; tiny-run val test reads metrics.jsonl.
- Surprises: squad_cloze_ratio is share of leftover `<full>` draws after cloze_ratio (matches squad_full_ratio). Memory chunk keeps padded first chunk (encoder needs chunk_size); EOS stripped only from question/target text. Overlapping answers merged; each distinct answer blanked at first occurrence. Val excludes paragraphs of answer_val[:max(40, val_recall_samples)] — assumes recall evals use first N of answer_val. Needs qa_train_samples / qa_val_samples set; new counts need network. No real-data run done.
- Built by: slice-implementer · sonnet · 36022 · verification passed first time

## Slice 02 — cloze-recall-eval
- Built: `scripts/eval.py --suite cloze` → `<run>/eval/cloze_recall.{json,md}` (real/shuffled/zeroed hit rate + answer loss, 40 distinct SQuAD dev paragraphs); trainer val exclusion now uses the exact eval paragraph set.
- Files touched: src/engram/eval/cloze_recall.py (new), scripts/eval.py, tests/test_cloze.py, src/engram/train/trainer.py, scripts/train.py
- New exports/interfaces: `select_recall_paragraphs(tokenizer, examples, max_len, n=40) -> list[(src_example, cloze_example)]`; `evaluate_cloze_recall(eng, cloze_examples, max_new_tokens=48, batch_size=8) -> {k, n, real|shuffled|zeroed: {hits, hit_rate, answer_loss}}`; `cloze_recall_markdown`, `write_cloze_recall_report`, `N_PARAGRAPHS=40`, `CLOZE_DEV_SAMPLES=3000`; `trainer.squad_cloze_val_examples(tokenizer, dev, max_len)`; `train(..., cloze_dev=None)` (scripts/train.py loads 3000 dev when squad_cloze_ratio > 0).
- Conventions confirmed: tests use `fact_tokenizer`, `qa_ex`; stub engine at decoder boundary only. Tiny test tokenizer maps most digits to unk — use vocab-safe fixtures.
- Surprises: Did not reproduce the old K=16 numbers; user re-baselined (see DECISIONS). Baseline K=16 ckpt-3000: real 3.48 / shuffled 4.60 / zeroed 4.21 nats, hits 5/3/11 of 40 — report at runs/fact_objective/k16/20261126-212058/eval/cloze_recall.json. Old val set was effectively empty.
- Built by: slice-implementer · sonnet · 51421 · verification passed first time (acceptance numbers re-baselined by user)
