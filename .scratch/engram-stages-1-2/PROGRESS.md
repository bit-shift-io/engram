# Progress — engram-stages-1-2

## Project mechanics
Test: UV_CACHE_DIR=$TMPDIR/uvcache HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 /private/tmp/claude-501/-Users-fabian-Projects-engram/f550dd86-93eb-4201-b90c-d2f8293b1a8d/scratchpad/uvenv/bin/uv run pytest -m "not slow" -q
Type-check: n/a
Lint: none configured

## Slice 01 — skeleton-and-base-model
- Built: uv project (torch 2.2.2, Py 3.12, numpy<2), YAML→dataclass config, frozen base-model loader with greedy generate, scripts/generate.py.
- Files touched: pyproject.toml, .python-version, .gitignore, uv.lock, configs/default.yaml, src/engram/__init__.py, src/engram/model/__init__.py, src/engram/config.py, src/engram/model/base.py, scripts/generate.py, tests/test_config.py, tests/test_base_model.py
- New exports/interfaces: engram.config.Config (frozen dataclass: model_name="HuggingFaceTB/SmolLM2-135M", device="cpu", chunk_size=128, k=16, seed=0, run_dir="runs"); load_config(path, overrides=None) -> Config (ValueError on unknown fields); engram.model.base.freeze(model); load_base_model(cfg) -> (model, tokenizer); generate(model, input_ids, device, max_new_tokens=20) -> new token ids
- Conventions confirmed: tests/test_*.py; real-weight tests @pytest.mark.slow; fast tests use tiny random-init LlamaForCausalLM; src/ layout, hatchling.
- Surprises: uv not installed system-wide — lives at scratchpad/uvenv/bin/uv (temporary; see Test line). Set UV_CACHE_DIR=$TMPDIR/uvcache, HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1. Sandbox blocks *.pem so HF downloads fail inside it; SmolLM2-135M weights now cached in $TMPDIR/hf.
- Built by: slice-implementer · inherited · 24064 · verification passed first time

## Slice 02 — encode-decode-roundtrip
- Built: Encoder (memory queries after chunk, encoder LoRA, Linear+RMSNorm projection), Decoder (target-only teacher-forced loss, greedy generate), Engram wrapper; tiny-model overfit ≥90%.
- Files touched: src/engram/tokens.py, src/engram/model/{memory,encoder,decoder,engram,base}.py, tests/{conftest,test_encoder,test_decoder,test_overfit}.py
- New exports/interfaces: engram.tokens CONTROL_TOKENS, add_control_tokens(tokenizer, model)->dict; base ADAPTERS, add_lora(model, r=8, alpha=16)->PeftModel, use_adapter(peft_model, name) (re-enables grads on all lora_ params); memory MemoryQueries, RMSNorm, MemoryProjection, ControlEmbedding; Encoder(model, chunk_size, k, d, embed_rms)(ids[B,N*C])->[B,N*K,d]; Decoder(model, control).loss(memory, control_ids, target_ids)/.generate(memory, control_ids, max_new_tokens, eos_id)->list[int]; Engram(base_model, tokenizer, cfg) .encode/.loss(chunk_ids, density="full") (target = chunk+EOS)/.generate(text, density, max_new_tokens)->str/.tokenize_chunks(text)->[1,N*C] EOS-padded/.trainable_parameters()/.control_ids/.hidden_size/.eos_id/.lm
- Conventions confirmed: tests/conftest.py tiny_engram(chunk_size, k) fixture (tiny Llama, WordLevel tokenizer).
- Surprises: real SmolLM2 weights NOT cached in sandbox $TMPDIR/hf — real-weight overfit test (slow, lr 1e-3 guess) never ran. Control-token trainable vectors live in separate ControlEmbedding; resized base rows stay frozen. LoRA r/alpha not in Config yet.
- Built by: slice-implementer · inherited · 33357 · verification passed first time
- Rework (verification on real weights): test chunk was 20% EOS padding (fixed + asserted); encoder now runs body only (get_base_model().model(...).last_hidden_state), ~1.2s/step; grad clipping 1.0 needed (lr 5e-3 diverges unclipped; 2e-3 clipped converges by step 40) — clipping is only in the test helper, training loop must add its own. Real overfit: acc 1.000. 5-min limit relaxed by user (Mac under load; 1698s loaded, 162s unloaded); time assert now opt-in via ENGRAM_OVERFIT_MAX_SECS. Weights at repo .hf/ (gitignored), HF_HUB_OFFLINE=1.
- Built by: slice-implementer · inherited · 45472 (rework) · needed 1 re-run

## Slice 03 — stage1-training
- Built: WikiText-103 streaming pipeline with on-disk token cache (data/cache/), Stage 1 trainer (grad accum, warmup+cosine, clip 1.0, periodic val, trainable-only ckpts, resume), scripts/train.py, configs/stage1.yaml + smoke.yaml.
- Files touched: src/engram/data/{__init__,chunking,reconstruction}.py, src/engram/train/{__init__,losses,trainer}.py, scripts/train.py, configs/{stage1,smoke}.yaml, tests/{test_chunking,test_trainer}.py; also config.py, pyproject.toml (datasets>=2.20,<3, pyarrow<17), .gitignore (data/cache/), uv.lock
- New exports/interfaces: chunk_tokens(ids, size) (tail dropped); clean_lines(lines); load_chunks(cfg, tokenizer, split, n)->LongTensor[n,chunk_size] (split "train"/"validation"); reconstruction_loss(eng, batch); validation_loss(eng, chunks, batch_size)->float; lr_at(step, cfg); save_checkpoint(eng, opt, step, path); load_checkpoint(eng, path, opt=None)->step; train(eng, cfg, train_chunks, val_chunks)->run_dir. Config fields: dataset, dataset_config, data_cache, train_samples, val_samples, lr, steps, batch_size, grad_accum, warmup_steps, max_grad_norm, val_interval, ckpt_interval, num_threads, resume. Ckpt = {step, params, opt} at runs/<ts>/ckpt-N.pt; run dir has config.yaml, metrics.jsonl (train rows: train_loss, lr, grad_norm, secs; val rows: val_loss). scripts/train.py --config X [--resume ckpt.pt]
- Conventions confirmed: trainer tests import make_tiny_model, make_tiny_tokenizer from conftest (no tests package). Drop HF_HUB_OFFLINE=1 for first dataset download. Smoke = wikitext-2, 40 train / 8 val chunks.
- Surprises: pyarrow pinned <17 to keep numpy<2. ~0.95 s/chunk; stage1 = 800 steps × 8 chunks ≈ 1.8–2 h idle. 30-step probe val loss 3.416→3.207→3.136. lr 2e-3 untuned. Probe output left in runs/probe/ (subagent rm denied).
- Built by: slice-implementer · sonnet · n/a (not reported) · verification passed first time

## Slice 04 — reconstruction-and-ablation-eval
- Built: reconstruction eval (token accuracy + ROUGE-L under real/zeroed/shuffled memory), per-run JSON + markdown report, multi-run K comparison table.
- Files touched: src/engram/eval/{__init__,ablation,reconstruct,report}.py, scripts/eval.py, tests/test_eval.py
- New exports/interfaces: zero_memory(m), shuffle_memory(m) (roll by 1 on batch dim; raises if batch<2), CONDITIONS; token_accuracy(pred, target); rouge_l(pred, target) (LCS F1 over token ids, no dependency); evaluate_reconstruction(eng, chunks, batch_size=8)->{n_chunks, chunk_size, k, compression_ratio, real|zeroed|shuffled:{token_accuracy, rouge_l}}; write_run_report(run_dir, result) -> <run>/eval/reconstruction.{json,md}; markdown_table(results); load_results(run_dirs). scripts/eval.py --run/--runs DIR... [--n N]; main(argv), latest_checkpoint(run_dir).
- Conventions confirmed: test_eval imports overfit from test_overfit; eval reads config.yaml + highest ckpt-N.pt; compression = chunk_size/k.
- Surprises: none. Verified here end-to-end: smoke train → scripts/eval.py on real weights produced the table (accuracies ~0 after 5 steps, as expected). Real ≥80% and K sweep await real Stage 1 runs.
- Built by: slice-implementer · sonnet · 24076 · verification passed first time

## Slice 05 — stage2-density-control
- Built: XSum <terse> pipeline (512 tokens → ≤4 EOS-padded chunks, masked padded memory), batch-level MixedSampler, per-group LR + init-from-checkpoint, configs/stage2.yaml, scripts/train.py --init-from.
- Files touched: src/engram/data/{compression,mixed}.py, src/engram/train/trainer.py, configs/stage2.yaml, tests/test_compression_data.py, src/engram/config.py; outside list: src/engram/model/{decoder,engram}.py, scripts/train.py, tests/test_trainer.py
- New exports/interfaces: make_example(tokenizer, article, summary, chunk_size, max_chunks, max_summary_tokens=None)->{chunks[n,C], target[T] ends EOS, density:"terse"}; load_terse_examples(cfg, tok, split, n); MixedSampler(full_chunks, terse_examples, full_ratio, batch_size, generator).next()->("full",[B,C])|("terse",list[dict]); Engram.terse_loss(examples); Decoder.loss(memory, control_ids, target_ids, memory_mask=None, target_mask=None); param_groups(eng, cfg) (groups carry `scale`); train(eng, cfg, train_chunks, val_chunks, terse_train=None, terse_val=None). Config: init_from, encoder_lr_scale(1.0), full_ratio(1.0), terse_dataset, terse_train_samples, terse_val_samples, max_chunks(4), max_summary_tokens(64).
- Conventions confirmed: cfg_for helper in test_trainer.py; encoder param group = encoder.* + LoRA names containing ".encoder.".
- Surprises: stage2 ≈ 2.8 h worst case (500 steps, grad_accum 2, full_ratio 0.5). Stage 1 sampling now via MixedSampler. Run dirs named by second collide if two start in the same second (pre-existing). Smoke2 config only in scratchpad. EdinburghNLP/xsum streams fine with datasets 2.21.
- Built by: slice-implementer · inherited · n/a (not reported) · verification passed first time

## Slice 06 — density-eval-and-demo
- Built: eval.py --suite density (mean <terse>/<full> length ratio, <terse> ROUGE-L beside frozen-base "TL;DR:" baseline), scripts/demo.py, KV cache in Decoder.generate.
- Files touched: src/engram/eval/density.py, scripts/eval.py, scripts/demo.py, tests/test_density_eval.py; outside list: src/engram/model/decoder.py (KV cache)
- New exports/interfaces: evaluate_density(eng, examples, max_full_tokens=None, max_terse_tokens=64); aggregate_density(records); baseline_summary(eng, article_ids, max_new_tokens); strip_trailing_eos(ids, eos_id); density_markdown(results); write_density_report(run_dir, result); scripts/demo.py "<text>" --run RUN [--max-full-tokens 256]
- Conventions confirmed: eval writes <run>/eval/<suite>.{json,md}.
- Surprises: demo needed a KV cache (was >400 s) plus a 256-token <full> cap to hit 30 s (23 s on 511 tokens); no cached-vs-uncached equivalence test; first density eval needs network for XSum validation.
- Built by: slice-implementer · sonnet · n/a (not reported) · verification passed first time
