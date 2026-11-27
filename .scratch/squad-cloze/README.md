# SQuAD gold-answer cloze

## Problem
- Memory keeps the gist of a paragraph but not its facts: answer recall is 7.5% on SQuAD dev.
- The WikiText cloze task built a 1.3-nat real-vs-shuffled gap on WikiText that did not carry over to SQuAD.
- On SQuAD dev, the fact K=16 memory lowers gold-answer loss by only 0.5 nats over zeroed memory.

## Solution
- Continue the fact K=16 checkpoint with a third task: the cloze task on SQuAD train paragraphs, blanking gold answer spans.
- Add a cloze recall eval on unseen SQuAD dev paragraphs: real vs shuffled vs zeroed memory.
- Run unattended, with an early stop if memory is not being used.

## Out of Scope
- K=32 (see `docs/memory/k32-fails-with-extra-objectives.md`).
- Changing the encoder, decoder or memory adapter.
- The joint `<terse>`/`<answer>` stage.

## Cross-cutting Constraints
- Train only on SQuAD train paragraphs; dev paragraphs stay unseen.
- SQuAD enters training only through cloze batches, never as `<full>` reconstruction chunks.
- No EOS padding in SQuAD cloze examples.
- Run settings: `.venv/bin/python`, `HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1`; data caches are keyed by sample count.

## Acceptance Criteria
- [ ] One command trains, evaluates and writes a summary with no human input.
- [ ] Summary reports answer recall and the cloze recall real-minus-zeroed answer-loss gap against last night's fact K=16.
- [ ] Pass = answer recall ≥ 0.15 and real beats zeroed by ≥ 1 nat; each reported separately.
- [ ] Run stops at step 1000 when SQuAD cloze real and zeroed val loss differ by < 0.1 nats.

## References
- Glossary: Cloze task, Fact weighting (`docs/CONTEXT.md`).
- `docs/memory/reconstruction-keeps-gist-not-facts.md`, `docs/memory/reconstruction-mix-dilutes-wikitext.md`.
- Start checkpoint: `runs/fact_objective/k16/20261126-212058/ckpt-3000.pt`.
