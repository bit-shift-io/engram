# Progress — cloze-eval-fix

## Project mechanics
Test: HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m "not slow" -q
Type-check: n/a
Lint: none configured

## Slice 01 — word-boundary-matching
- Built: shared word-boundary matcher used by `answer_in_text` and `make_gold_cloze_example`.
- Files touched: src/engram/eval/qa.py, src/engram/eval/recall.py, src/engram/data/cloze.py, tests/test_recall_eval.py, tests/test_cloze.py
- New exports/interfaces: `engram.eval.qa.find_word_bounded(needle: str, text: str) -> int` returns the first word-bounded index, or -1.
- Conventions confirmed: tests live in `tests/test_*.py`; `answer_in_text` is tested in `test_recall_eval.py`.
- Surprises: `data/cloze.py` now imports `engram.eval.qa` (no cycle). Blanker matches case-sensitively on raw text; `answer_in_text` matches on normalised text. Tests written after code, not red-first.
- Built by: slice-implementer · sonnet · 17064 · verification passed first time

## Slice 02 — cross-article-shuffle
- Built: SQuAD examples carry `title` (cache key `-ans-t`); cloze recall's shuffled memory (generation and answer loss) always comes from a different article.
- Files touched: src/engram/data/qa.py, src/engram/eval/cloze_recall.py, tests/test_cloze.py
- New exports/interfaces: `engram.eval.cloze_recall.cross_article_partners(titles: list[str]) -> list[int]`; `evaluate_cloze_recall` requires `"title"` on each cloze example; `load_qa_examples` returns dicts with `"title"`.
- Conventions confirmed: tests use `fact_tokenizer()` and the `StubEng` fixture in tests/test_cloze.py; `datasets` is faked through `sys.modules` for loader tests.
- Surprises: key bump applies to all splits, so train scripts need a one-time SQuAD re-download before offline runs. Dev cache not built yet (needs the user to download with network). Old `-ans` caches lack title. Shuffled loss assumes each cloze example is a single chunk.
- Built by: slice-implementer · inherited model · 32451 · verification passed first time

## Slice 03 — blank-recall-scoring
- Built: per-blank answers in gold cloze examples; cloze recall and aligned recall per condition in cloze_recall.{json,md}, replacing the hit count.
- Files touched: src/engram/data/cloze.py, src/engram/eval/cloze_recall.py, tests/test_cloze.py
- New exports/interfaces: `engram.eval.cloze_recall.score_blanks(blanks: list[list[str]], text: str) -> tuple[int, int]` (recalled, aligned). Gold cloze examples gain `blanks: list[list[str]]`; `evaluate_cloze_recall` examples require `"blanks"` and `"title"`. Result per condition: `{blanks, cloze_recall, aligned_recall, answer_loss}` (micro-averaged over blanks).
- Conventions confirmed: tests use `fact_tokenizer()`, `StubEng` and `qa_ex` in tests/test_cloze.py.
- Surprises: `src/engram/eval/report.py` `squad_cloze_markdown.row()` (~line 226) still reads `cl[k]["hits"]` → KeyError on new output; left for slice 04. Old results lacking `blanks` cannot be rescored. Follow-up (inline, user choice): SQuAD cache key `-ans-t` now applies to the validation split only; train keeps `-ans`.
- Built by: slice-implementer · sonnet · 27733 · verification passed first time

## Slice 04 — rescore-checkpoints
- Built: `scripts/rescore_cloze.py` re-scores both checkpoints into `<run>/eval_rescore/` and writes `runs/cloze_rescore/summary.{json,md}`; relative 2x pass rule flagged per condition; summary markdown also shows cloze/aligned recall per condition.
- Files touched: scripts/rescore_cloze.py, scripts/eval.py (`--eval-dir`, shuffle pool), src/engram/eval/report.py, src/engram/eval/recall.py, src/engram/eval/cloze_recall.py (`eval_dir`, `shuffle_pool`, `other_article_pool`), src/engram/data/qa.py (`with_title`), scripts/train.py, tests/test_squad_cloze_pipeline.py, tests/test_cloze.py
- New exports/interfaces: `report.rescore_summary`, `report.rescore_markdown`, `rescore_cloze.run_rescore`, `rescore_cloze.old_scores`; `load_qa_examples(..., with_title=False)`; `evaluate_cloze_recall(..., shuffle_pool=None)`; `other_article_pool(examples, exclude_titles, n)`.
- Conventions confirmed: script tests load scripts via importlib with stub runners.
- Surprises: real run failed twice. (1) Slice 02's validation-wide cache rename invalidated the 200-example answer-recall and training val caches → now opt-in `with_title`. (2) All 40 selected eval paragraphs are one article → shuffled memory borrowed from 40 other-article dev paragraphs (user choice), eval set unchanged. Result: PASS — answer recall 0.175 vs 0.025 baseline, gap 1.76 vs 0.73.
- Built by: slice-implementer · sonnet · 29363 · needed 2 re-runs (fixed inline)
