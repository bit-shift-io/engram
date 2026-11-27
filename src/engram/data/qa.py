"""SQuAD v1.1 examples for Stage 3 `<answer>` training: paragraph -> <=max_chunks chunks (longer
paragraphs are skipped), decoder prompt = question, target = answer + EOS."""

import hashlib
from pathlib import Path

import torch

from engram.config import Config


def make_qa_example(tokenizer, context: str, question: str, answer: str, chunk_size: int,
                    max_chunks: int,
                    answers: list[str] | None = None) -> dict | None:
    """None if the paragraph needs more than max_chunks chunks. `answers` = all gold answer texts
    (kept as "answers" for EM/F1; defaults to [answer])."""
    eos = tokenizer.eos_token_id
    ids = tokenizer(context, add_special_tokens=False).input_ids or [eos]
    if len(ids) > chunk_size * max_chunks:
        return None
    ids = ids + [eos] * (-len(ids) % chunk_size)
    q = tokenizer(question, add_special_tokens=False).input_ids
    a = tokenizer(answer, add_special_tokens=False).input_ids
    return {
        "chunks": torch.tensor(ids).reshape(-1, chunk_size),
        "question": torch.tensor(q),
        "target": torch.tensor(a + [eos]),
        "density": "answer",
        "answers": list(answers) if answers else [answer],
    }


def load_qa_examples(cfg: Config, tokenizer, split: str, n: int, with_title: bool = False) -> list[dict]:
    """n SQuAD examples (first answer) for a split, cached under cfg.data_cache.

    with_title=True guarantees each example has the article "title" (cloze recall's cross-article
    shuffle needs it) and uses its own cache key; other callers keep their existing caches.
    """
    suffix = "-ans-t" if with_title else "-ans"
    key = f"{cfg.qa_dataset}-{split}-{cfg.model_name}-{cfg.chunk_size}-{cfg.max_chunks}-{n}{suffix}"
    path = Path(cfg.data_cache) / (hashlib.md5(key.encode()).hexdigest()[:12] + ".pt")
    if path.exists():
        return torch.load(path)
    from datasets import load_dataset

    ds = load_dataset(cfg.qa_dataset, split=split, streaming=True)
    out = []
    for r in ds:
        answers = r["answers"]["text"]
        if not r["context"].strip() or not answers:
            continue
        ex = make_qa_example(tokenizer, r["context"], r["question"], answers[0],
                             cfg.chunk_size, cfg.max_chunks, answers=answers)
        if ex is not None:
            out.append({**ex, "title": r["title"]})
        if len(out) >= n:
            break
    if len(out) < n:
        raise ValueError(f"only {len(out)} examples available for {split}, wanted {n}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out
