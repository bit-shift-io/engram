"""XSum examples for Stage 2 `<terse>` training: article -> <=max_chunks chunks, target = summary + EOS."""

import hashlib
from pathlib import Path

import torch

from engram.config import Config


def make_example(tokenizer, article: str, summary: str, chunk_size: int, max_chunks: int,
                 max_summary_tokens: int | None = None) -> dict:
    """Truncate the article to chunk_size*max_chunks tokens and EOS-pad it to whole chunks."""
    eos = tokenizer.eos_token_id
    ids = tokenizer(article, add_special_tokens=False).input_ids[: chunk_size * max_chunks]
    ids = ids or [eos]
    ids = ids + [eos] * (-len(ids) % chunk_size)
    tgt = tokenizer(summary, add_special_tokens=False).input_ids
    if max_summary_tokens:
        tgt = tgt[:max_summary_tokens]
    return {
        "chunks": torch.tensor(ids).reshape(-1, chunk_size),
        "target": torch.tensor(tgt + [eos]),
        "density": "terse",
    }


def load_terse_examples(cfg: Config, tokenizer, split: str, n: int) -> list[dict]:
    """n XSum examples for a split, cached under cfg.data_cache."""
    key = (f"{cfg.terse_dataset}-{split}-{cfg.model_name}-{cfg.chunk_size}-{cfg.max_chunks}"
           f"-{cfg.max_summary_tokens}-{n}")
    path = Path(cfg.data_cache) / (hashlib.md5(key.encode()).hexdigest()[:12] + ".pt")
    if path.exists():
        return torch.load(path)
    from datasets import load_dataset

    ds = load_dataset(cfg.terse_dataset, split=split, streaming=True)
    out = []
    for r in ds:
        if not r["document"].strip() or not r["summary"].strip():
            continue
        out.append(make_example(tokenizer, r["document"], r["summary"], cfg.chunk_size,
                                cfg.max_chunks, cfg.max_summary_tokens))
        if len(out) >= n:
            break
    if len(out) < n:
        raise ValueError(f"only {len(out)} examples available for {split}, wanted {n}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out
