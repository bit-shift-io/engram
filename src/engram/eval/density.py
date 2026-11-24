"""Density eval: <full> vs <terse> output length on one memory, <terse> ROUGE-L vs a TL;DR baseline."""

import json
from pathlib import Path

import torch

from engram.eval.reconstruct import rouge_l
from engram.model.base import generate


def strip_trailing_eos(ids: list[int], eos_id: int) -> list[int]:
    ids = list(ids)
    while ids and ids[-1] == eos_id:
        ids.pop()
    return ids


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def aggregate_density(records: list[dict]) -> dict:
    ratios = [r["terse_len"] / r["full_len"] for r in records if r["full_len"] > 0]
    return {
        "n": len(records),
        "mean_full_len": _mean([r["full_len"] for r in records]),
        "mean_terse_len": _mean([r["terse_len"] for r in records]),
        "length_ratio": _mean(ratios),
        "terse_rouge_l": _mean([r["terse_rouge_l"] for r in records]),
        "baseline_rouge_l": _mean([r["baseline_rouge_l"] for r in records]),
    }


@torch.no_grad()
def baseline_summary(eng, article_ids: list[int], max_new_tokens: int) -> list[int]:
    """Frozen base model (LoRA adapters disabled) prompted with 'TL;DR:'."""
    prompt = article_ids + eng.tokenizer("\nTL;DR:", add_special_tokens=False).input_ids
    with eng.lm.disable_adapter():
        out = generate(eng.lm, torch.tensor([prompt]), eng.cfg.device, max_new_tokens)
    return strip_trailing_eos(out[0].tolist(), eng.eos_id)[:max_new_tokens]


@torch.no_grad()
def evaluate_density(eng, examples: list[dict], max_full_tokens: int | None = None,
                     max_terse_tokens: int = 64) -> dict:
    eng.eval()
    k = eng.cfg.k
    records = []
    for ex in examples:
        flat = ex["chunks"].reshape(1, -1)
        memory = eng.encode(flat)
        n_full = max_full_tokens or memory.shape[1] // k * eng.cfg.chunk_size + 1
        full = eng.decoder.generate(memory, eng._control("full", 1), n_full, eng.eos_id)
        terse = eng.decoder.generate(memory, eng._control("terse", 1), max_terse_tokens, eng.eos_id)
        ref = ex["target"].tolist()[:-1]
        base = baseline_summary(eng, strip_trailing_eos(flat[0].tolist(), eng.eos_id), max_terse_tokens)
        records.append({"full_len": len(full), "terse_len": len(terse),
                        "terse_rouge_l": rouge_l(terse, ref), "baseline_rouge_l": rouge_l(base, ref)})
    return {"k": k, "compression_ratio": eng.cfg.chunk_size / k, **aggregate_density(records)}


def density_markdown(results: list[dict]) -> str:
    rows = ["| K | n | mean <full> len | mean <terse> len | terse/full ratio | <terse> ROUGE-L | base TL;DR ROUGE-L |",
            "|---|---|---|---|---|---|---|"]
    for r in results:
        rows.append(f"| {r['k']} | {r['n']} | {r['mean_full_len']:.1f} | {r['mean_terse_len']:.1f} | "
                    f"{r['length_ratio']:.3f} | {r['terse_rouge_l']:.3f} | {r['baseline_rouge_l']:.3f} |")
    return "\n".join(rows)


def write_density_report(run_dir, result: dict) -> Path:
    d = Path(run_dir) / "eval"
    d.mkdir(exist_ok=True)
    (d / "density.json").write_text(json.dumps(result, indent=2))
    (d / "density.md").write_text(density_markdown([result]) + "\n")
    return d / "density.json"
