# Fact-preserving memory objective

## Problem
- Plain reconstruction keeps the gist of a chunk but drops names, numbers and dates.
- Answer recall from `<full>` is 7.5% (K=16) and 5% (K=32); the Stage 3b recall gate needs 25%.
- Raising K alone did not help. The objective does not reward facts.

## Solution
- Replace plain reconstruction with fact weighting (70% of steps) plus a cloze task (30%).
- One unattended run: fresh K=16 → evals → if it fails, fresh K=32 → evals → summary.
- Pass = answer recall ≥ 15% on 40 SQuAD dev paragraphs, with real memory beating shuffled.

## Out of Scope
- The joint `<full>`/`<terse>`/`<answer>` stage. Re-run Stage 3b only after a pass.
- SQuAD paragraphs in training (see docs/memory/reconstruction-mix-dilutes-wikitext.md).
- NER models or new dependencies for fact detection.
- K=64.

## Cross-cutting Constraints
- Stage 1 settings everywhere except the objective: WikiText only, 2000 train chunks, lr 0.002, batch 4, frozen decoder, memory adapter rank 8.
- Never load or overwrite existing runs. New output goes under `runs/fact_objective/`.
- Evaluate real / shuffled / zeroed memory (docs/memory/memory-necessity-eval.md). Fact weights come from the frozen decoder with zeroed memory.
- Default config values keep current behaviour; existing tests must pass unchanged.
- Whole run unattended. Each stage resumes from its last checkpoint.

## Acceptance Criteria
- [ ] Unattended run: K=16 → recall + memory-use evals → K=32 only on fail → summary.
- [ ] Summary reports answer recall, recon token F1, real/shuffled/zeroed ROUGE-L per arm, against K=16 Stage 1 and K=32 WikiText-only baselines.
- [ ] Fast tests pass.

## References
- docs/memory/reconstruction-keeps-gist-not-facts.md, reconstruction-mix-dilutes-wikitext.md, decoder-ignores-memory.md, memory-necessity-eval.md
- Baselines: runs/20261125-090655 (K=16 Stage 1), runs/k32_wikionly/20261126-164326 (K=32, recall 0.050)
- configs/stage1.yaml, scripts/stage3b.py (pipeline pattern)
