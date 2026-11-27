---
name: cloze-eval-paragraphs-one-article
description: Cloze/answer recall eval paragraphs used to be one SQuAD article (dataset order) and answer recall only 2 paragraphs; now 40 paragraphs round-robin over articles; SQuAD cache keys must not change silently
metadata:
  type: convention
---

**Why:** SQuAD dev is grouped by article. Taking the first 40 qualifying paragraphs gave 40 from Super_Bowl_50, and answer recall's first 40 questions covered only 2 paragraphs. Results described one topic and were easy to guess from repeated names (fixed 2026-11-28).
**How to apply:** When changing the eval paragraph selection, comparing recall numbers across dates, or touching `load_qa_examples` cache keys.

- `select_recall_paragraphs` takes articles round-robin (4 each from the 10 in the 3,000-example dev cache). The trainer excludes the same paragraphs from its validation loss; dev is never trained on, so changing the selection is safe.
- Answer recall and cloze recall score the same 40 paragraphs. Answer recall uses one question per paragraph (its first); pooling all its questions' answers inflates the score.
- Shuffled memory comes from another article; `other_article_pool` lends memory from unscored paragraphs if the set ever spans one article.
- Numbers from before 2026-11-28 are on the old selection and don't compare with later ones.
- Changing a `load_qa_examples` cache key forces a download, which breaks offline runs. Add new fields behind an opt-in flag with its own key (as `with_title=True` does).
