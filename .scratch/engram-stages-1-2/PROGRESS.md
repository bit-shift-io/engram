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
