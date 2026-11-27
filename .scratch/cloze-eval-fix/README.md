# Cloze eval fix

## Problem
- Gold-answer matching uses substrings: "L" blanks "Football ___eague", "5" blanks "Super Bowl ___0".
- The same matcher scores answer recall and cloze hits, so single-digit gold answers make most outputs a hit (zeroed 13/40).
- Shuffled memory comes from the previous paragraph, usually the same article, so it leaks facts.

## Solution
- Match gold answers on word boundaries everywhere: blanker, answer recall, cloze recall.
- Score cloze recall per blanked answer, order-free, plus aligned recall; shuffle memory across articles.
- Re-score the fact K=16 baseline and the SQuAD cloze checkpoint; pass = at least 2× the re-scored baseline.

## Out of Scope
- Retraining. The fixed blanker applies to future training runs only.
- Changing the model, the training mix or the 40-paragraph eval set size.

## Cross-cutting Constraints
- Run settings: `.venv/bin/python`, `HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1`; data caches are keyed by sample count.
- Dev paragraphs stay out of training; the eval paragraph selection must stay the one the trainer excludes.

## Acceptance Criteria
- [ ] No gold answer matches inside a word, in the blanker or either eval.
- [ ] Shuffled memory always comes from a different article.
- [ ] One command re-scores both checkpoints and writes a summary.
- [ ] Summary reports answer recall, cloze recall, aligned recall and the real-minus-zeroed loss gap for both checkpoints.
- [ ] Pass = answer recall and loss gap each ≥ 2× the re-scored baseline, reported separately.

## References
- Glossary: Cloze recall, Aligned recall, Answer recall (`docs/CONTEXT.md`).
- `docs/memory/reconstruction-keeps-gist-not-facts.md`, `docs/memory/hf-weights-and-sandbox.md`.
- Checkpoints: `runs/fact_objective/k16/20261126-212058` (baseline), `runs/squad_cloze/train/20261127-163240` (new).
