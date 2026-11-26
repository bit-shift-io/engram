"""WikiText chunks for Stage 1 reconstruction, cached tokenized on disk."""

import hashlib
from pathlib import Path

import torch

from engram.config import Config
from engram.data.chunking import chunk_tokens


def clean_lines(lines):
    """Drop blanks and ' = Heading = ' lines; strip whitespace."""
    for line in lines:
        t = line.strip()
        if t and not (t.startswith("=") and t.endswith("=")):
            yield t


def _stream_chunks(cfg: Config, tokenizer, split: str, n: int) -> list[list[int]]:
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset, cfg.dataset_config, split=split, streaming=True)
    buf: list[int] = []
    chunks: list[list[int]] = []
    for text in clean_lines(r["text"] for r in ds):
        buf += tokenizer(text, add_special_tokens=False).input_ids
        full = chunk_tokens(buf, cfg.chunk_size)
        if full:
            chunks += full
            buf = buf[len(full) * cfg.chunk_size :]
        if len(chunks) >= n:
            break
    return chunks[:n]


def load_chunks(cfg: Config, tokenizer, split: str, n: int) -> torch.Tensor:
    """[n, chunk_size] token ids for a split ('train'/'validation'), cached under cfg.data_cache."""
    key = f"{cfg.dataset}-{cfg.dataset_config}-{split}-{cfg.model_name}-{cfg.chunk_size}-{n}"
    path = Path(cfg.data_cache) / (hashlib.md5(key.encode()).hexdigest()[:12] + ".pt")
    if path.exists():
        return torch.load(path)
    chunks = _stream_chunks(cfg, tokenizer, split, n)
    if len(chunks) < n:
        raise ValueError(f"only {len(chunks)} chunks available for {split}, wanted {n}")
    t = torch.tensor(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t, path)
    return t


def squad_paragraph_chunks(examples: list[dict], n: int) -> torch.Tensor:
    """[n, chunk_size] chunks from distinct SQuAD paragraphs (examples repeat a paragraph per question)."""
    seen, out = set(), []
    for ex in examples:
        key = tuple(ex["chunks"].flatten().tolist())
        if key in seen:
            continue
        seen.add(key)
        out += list(ex["chunks"])
        if len(out) >= n:
            break
    if len(out) < n:
        raise ValueError(f"only {len(out)} SQuAD paragraph chunks available, wanted {n}")
    return torch.stack(out[:n])


def mix_chunks(wiki: torch.Tensor, squad_examples: list[dict], share: float, seed: int = 0) -> torch.Tensor:
    """len(wiki) chunks: round(share * len) from SQuAD paragraphs, the rest WikiText, shuffled."""
    n = len(wiki)
    n_sq = round(share * n)
    if n_sq == 0:
        return wiki
    mixed = torch.cat([wiki[: n - n_sq], squad_paragraph_chunks(squad_examples, n_sq)])
    g = torch.Generator().manual_seed(seed)
    return mixed[torch.randperm(n, generator=g)]
