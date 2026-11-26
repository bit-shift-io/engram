# Progress — stage3-qa

## Project mechanics
Test: HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 .venv/bin/pytest -m "not slow" -q
Type-check: n/a
Lint: none configured

## Slice 01 — answer-task-tracer
- Built: `<answer>` control token, SQuAD QA examples, answer-only loss, MixedSampler answer share, trainer support, Stage 2 checkpoint expand-on-load; smoke run from ckpt-600 works (5 steps, finite losses).
- Files touched: src/engram/tokens.py, src/engram/data/qa.py (new), src/engram/data/mixed.py, src/engram/model/decoder.py, src/engram/model/engram.py, src/engram/train/trainer.py, src/engram/config.py, configs/smoke.yaml, scripts/train.py (outside list), tests/test_qa_data.py (new), tests/test_control_vectors.py, tests/test_trainer.py
- New exports/interfaces:
  - `engram.data.qa.make_qa_example(tokenizer, context, question, answer, chunk_size, max_chunks) -> dict | None` → `{chunks [n,C], question [Q], target [A+EOS], density: "answer"}`; None if > max_chunks.
  - `engram.data.qa.load_qa_examples(cfg, tokenizer, split, n) -> list[dict]` reads `cfg.qa_dataset` (default "rajpurkar/squad"), first answer, cached under `cfg.data_cache`.
  - `Decoder.loss(..., label_mask=None)`: 0 = attended but unsupervised.
  - `Engram.terse_loss(examples)` handles examples with "question" (prepended, unsupervised) — use for `<answer>` loss/val.
  - `MixedSampler(..., answer=None, answer_ratio=0.0)` may return `("answer", list)`. answer_ratio is share of all micro-batches; full_ratio splits the rest. 50/25/25 = answer_ratio 0.5, full_ratio 0.5, multi_full_ratio 0.5.
  - `train(..., answer_train=None, answer_val=None)` logs `n_answer`, `answer_val_loss`.
  - Config fields: `answer_ratio`, `qa_dataset`, `qa_train_samples`, `qa_val_samples`.
  - `load_checkpoint` grows `control.weight` / `decoder.ctrl_vec.vec` when checkpoint has fewer rows (`trainer._CONTROL_ROWS`); new control-embedding row keeps random init.
- Conventions confirmed: tests use `tiny_engram` / `make_tiny_tokenizer` from tests/conftest.py; Stage 2-shaped checkpoint fixture = slice saved control rows to `[:2]`.
- Surprises: `<answer>` generation (question in prompt) not built — `Engram.generate`/`Decoder.generate` take no question. smoke.yaml rewritten as Stage 3 smoke config (Stage 2 shape + 50/25/25 mix). No configs/stage3.yaml yet. SQuAD now cached under data/cache. Smoke run drew no `<terse>` batch; terse replay covered only by tiny-model test.
- Built by: slice-implementer · opus-5.5 · 56705 · verification passed first time

## Slice 02 — qa-eval
- Built: `eval.py --qa` scores EM/F1 under real/shuffled/zeroed memory plus zero-shot base model; writes `<run>/eval/qa.{json,md}`. Smoke-verified on runs/20261125-212227 with --n 3.
- Files touched: src/engram/eval/qa.py (new), tests/test_qa_eval.py (new), scripts/eval.py, src/engram/model/decoder.py, src/engram/data/qa.py, src/engram/config.py. report.py unchanged (QA table lives in eval/qa.py, density.py pattern).
- New exports/interfaces:
  - `Decoder.generate(memory, control_ids, max_new_tokens, eos_id, question_ids=None)` — question `[1,Q]` after control token; control vector covers it.
  - `make_qa_example(..., answers=None)` adds `ex["answers"]` (all gold texts). `load_qa_examples` cache key suffixed `-ans`.
  - `engram.eval.qa`: `normalize`, `exact_match(pred, golds)`, `f1_score(pred, golds)`, `evaluate_qa(eng, examples, max_new_tokens=32, zero_shot=True, seed=None)` → `{k, n, seed, real, shuffled, zeroed, base_zero_shot}` each `{em, f1}`; `qa_markdown`, `write_qa_report`.
  - Config: `qa_eval_samples=200`, `qa_eval_seed=0`. CLI: `eval.py --run <dir> --qa [--n N]`.
- Conventions confirmed: eval modules = result dict + markdown fn + writer. Zero-shot prompt `paragraph\nQuestion: q\nAnswer:` with adapters disabled; cut at EOS/newline, cap 32.
- Surprises: eval subset = first N dev examples that fit 4 chunks; seed only sets the shuffled derangement. Training answer_val also uses first dev examples → overlaps eval set. Loading SQuAD after a cache-key change needs network + sandbox off (HF_HUB_OFFLINE must be unset).
- Built by: slice-implementer · sonnet · 32556 · verification passed first time

## Slice 03 — text-qa-control
- Built: text-QA control (rank-8 LoRA on frozen base, plain `paragraph\nQuestion: q\nAnswer:` + answer + EOS, context masked), scored with slice 02 scorer, cached; QA report shows memory ÷ control F1. Only a 3-step smoke run done.
- Files touched: src/engram/eval/qa.py, src/engram/eval/report.py, src/engram/config.py, scripts/eval.py (one line), scripts/train_text_qa.py (new), configs/text_qa.yaml (new), tests/test_qa_eval.py
- New exports/interfaces:
  - `engram.eval.qa`: `text_qa_prompt(tok, ex, eos_id)`, `text_qa_example(tok, ex, eos_id) -> (ids, labels)`, `train_text_qa(model, tok, cfg, examples, run_dir) -> (peft_lm, losses)`, `text_qa_answer(lm, tok, ex, device, max_new_tokens)`, `evaluate_text_qa(lm, tok, examples, device, max_new_tokens=32) -> {n, em, f1}`; `qa_markdown(results, control=None)`, `write_qa_report(run_dir, result, control=None)`.
  - `engram.eval.report.load_control_result(path) -> dict | None`.
  - Config: `text_qa_steps: int = 0`, `text_qa_result: str = "runs/text_qa_control.json"`.
  - CLI: `scripts/train_text_qa.py --config configs/text_qa.yaml [--force] [--steps N] [--n N]` → runs/text-qa-<ts>/{config.yaml, ckpt-<steps>.pt, metrics.jsonl, eval/text_qa.json} + cache file; skips if cache exists unless --force.
- Conventions confirmed: zero-shot and control share `text_qa_prompt`/`text_qa_answer` (same layout and EOS/newline cut).
- Surprises: configs/text_qa.yaml has placeholder `text_qa_steps: 400` (0.5 × 800), batch_size 4, grad_accum 2, qa_train_samples 3200 — slice 04 must set text_qa_steps = answer_ratio × steps and copy batch_size/grad_accum/lr/LoRA from stage3.yaml. Smoke left runs/text-qa-20261125-212946 (deletable); cache written to scratchpad, not runs/.
- Built by: slice-implementer · opus-5.5 · 42813 · verification passed first time

## Slice 04 — stage3-run (code done; overnight run pending — Status stays in-progress)
- Built: configs/stage3.yaml (640 steps, ~3.6 h est.), trainer validation logging QA EM/F1 ×3 memory conditions + `<full>` gap + density; `eval/stage3.{json,md}` with Stage 2 regression gates and K=32 recommendation; text_qa_steps=320; Stage 3 how-to-run in docs/training.md.
- Files touched: configs/stage3.yaml (new), configs/text_qa.yaml, src/engram/train/trainer.py, src/engram/eval/report.py, src/engram/config.py, tests/test_trainer.py, docs/training.md, docs/CONTEXT.md
- New exports/interfaces: `report.STAGE2_GATES` (name -> (baseline, lo, hi)), `K32_THRESHOLD=0.5`, `stage3_gates(row, control) -> {metrics, regressions, memory_over_control_f1, recommend_k32}`, `stage3_markdown(g)`. Config: `val_gen_samples` (0=off), `val_max_full_tokens`. metrics.jsonl val keys: `qa_{real,shuffled,zeroed}_{em,f1}`, `full_gap`, `length_ratio`, `terse_article_ratio`, `full_article_ratio`, `terse_rouge_l`, `baseline_rouge_l`.
- Timing: answer/1-chunk full ~4 s, terse 6–8 s, whole-article full 20–28 s per micro-batch; ~14 s/step; val ~6 min.
- Surprises: gate bands chosen by implementer (gap ≥0.9; terse/article 0.05–0.15; full/article 0.80–1.05; ROUGE-L ≥0.164) — need confirming. Smoke flagged full_article_ratio 0.675 on 8 articles (likely noise). Shuffled F1 = real F1 in smoke. stage3 report needs the control result to exist first. SQuAD cached 3200 train / 200 val.
- Remaining: run control, then overnight Stage 3, then eval --qa / --suite density, record results in docs/training.md + docs/memory.
- Built by: slice-implementer · opus-5.5 · 58987 · verification passed first time

## Stage 3 outcome (2026-11-26) — negative; slice 04 left in-progress
- Overnight run runs/20261126-072712 + control runs/text-qa-20261126-064050 done. Memory F1 0.051 vs shuffled 0.048, control 0.687, ratio 0.074.
- Diagnostic: Stage 2 memory keeps answer in <full> recon 7.5% (token F1 0.26); Stage 3 <full> on SQuAD collapses to a short answer ("1999").
- Recorded in docs/training.md (Stage 3 results) and docs/memory/prefix-control-token-ignored.md.
- Decision: re-plan via plan-feature — K=32, short reconstruction warm-up, then joint <full>/<terse>/<answer> with <full> replay on SQuAD; add SQuAD <full> and answer-in-recon gates.
