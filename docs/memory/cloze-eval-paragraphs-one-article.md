---
name: cloze-eval-paragraphs-one-article
description: The 40 cloze/recall eval paragraphs are all from SQuAD's Super_Bowl_50 article (dataset order); shuffled memory is borrowed from other articles; SQuAD cache keys must not change silently
metadata:
  type: context
---

**Why:** `select_recall_paragraphs` takes the first 40 qualifying dev paragraphs in dataset order, and SQuAD dev is grouped by article, so all 40 come from Super_Bowl_50 (810 of the 3,000 cached dev questions). Results describe one topic with heavily repeated names, and a within-set shuffle has no other article to draw on (2026-11-28).
**How to apply:** When reading answer recall or cloze recall numbers, changing the eval paragraph selection, or touching `load_qa_examples` cache keys.

- Shuffled memory for cloze recall comes from `other_article_pool`: up to 40 dev paragraphs from other articles that lend memory but are not scored.
- Spreading the 40 across articles is safe: the trainer uses dev only for validation loss (`squad_cloze_val_examples`), never for training, so no checkpoint has seen dev paragraphs. Not yet done.
- Changing a `load_qa_examples` cache key forces a download, which breaks offline runs. Add new fields behind an opt-in flag with its own key (as `with_title=True` does) instead of changing the key for a whole split.
