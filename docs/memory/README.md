# Repo memory

- [intel-mac-torch-pin](intel-mac-torch-pin.md) — Intel Mac, CPU only: torch 2.2.2, Python 3.12, numpy<2, pyarrow<17
- [hf-weights-and-sandbox](hf-weights-and-sandbox.md) — HF_HOME=.hf in repo, offline tests, downloads need sandbox off (certifi .pem blocked)
- [peft-multi-adapter-grads](peft-multi-adapter-grads.md) — use use_adapter(), not set_adapter (freezes the other LoRA); clip grads at 1.0
- [decoder-ignores-memory](decoder-ignores-memory.md) — decoder LoRA shortcut ignores memory: train frozen + memory-only k/v adapter; LoRA later at low LR
- [prefix-control-token-ignored](prefix-control-token-ignored.md) — one <full>/<terse> token gets overridden; use control_vectors, a balanced mix, and replay old modes on each new mode's input type (Stage 3 <full> collapsed on SQuAD)
- [memory-necessity-eval](memory-necessity-eval.md) — judge memory by real vs shuffled vs zeroed on held-out chunks, not by loss
- [reconstruction-mix-dilutes-wikitext](reconstruction-mix-dilutes-wikitext.md) — adding SQuAD chunks by replacing WikiText (plus EOS padding) stalled K=32 warm-up; add on top, strip padding; K=32 plateaus ~1200 steps
- [reconstruction-keeps-gist-not-facts](reconstruction-keeps-gist-not-facts.md) — reconstruction keeps gist not facts; SQuAD gold-answer cloze stores facts (cloze recall 0.17 vs 0.06 baseline) but `<full>` does not write them out (answer recall 0.175 both); fix decoder side next
- [k32-fails-with-extra-objectives](k32-fails-with-extra-objectives.md) — fresh K=32 stalls (real = shuffled) with SQuAD mix or fact objectives; only plain WikiText K=32 learns; prefer K=16
- [cloze-eval-paragraphs-one-article](cloze-eval-paragraphs-one-article.md) — eval uses 40 paragraphs round-robin over 10 SQuAD articles, one question each for answer recall; pre-2026-11-28 numbers do not compare; never change SQuAD cache keys for a whole split
- [fact-qa-stages-result](fact-qa-stages-result.md) — Stage A+B from SQuAD cloze ckpt: facts kept (gate pass, recall 0.133) but QA only 0.26× control, `<full>` collapsed to terse length
- [smoke-runs-and-eval-sizes](smoke-runs-and-eval-sizes.md) — smoke shrinks steps and eval n only to cached sizes (QA 40/200); runs_smoke_*/ is gitignored; in-training QA checks too small to judge
