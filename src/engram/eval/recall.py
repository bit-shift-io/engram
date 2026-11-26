"""Answer recall: does a `<full>` reconstruction of a SQuAD paragraph still contain the gold
answer span? Also reconstruction token F1 and length ratio (reconstruction / paragraph tokens)."""

import json
from pathlib import Path

import torch

from engram.eval.qa import f1_score, normalize


def answer_in_text(answers: list[str], text: str) -> bool:
    """True if any non-empty normalised gold answer is a substring of the normalised text."""
    t = normalize(text)
    return any(a and a in t for a in map(normalize, answers))


@torch.no_grad()
def evaluate_recall(eng, examples: list[dict]) -> dict:
    """examples need chunks and answers. Generation goes through Engram.generate (memory hooks on)."""
    eng.eval()
    tok = eng.tokenizer
    hits, f1s, ratios = 0, [], []
    for ex in examples:
        ids = ex["chunks"].flatten().tolist()
        ctx = tok.decode(ids, skip_special_tokens=True)
        rec = eng.generate(ctx, "full")
        hits += answer_in_text(ex["answers"], rec)
        f1s.append(f1_score(rec, [ctx]))
        n_ctx = len(tok(ctx, add_special_tokens=False).input_ids)
        n_rec = len(tok(rec, add_special_tokens=False).input_ids)
        ratios.append(n_rec / max(n_ctx, 1))
    n = len(examples)
    return {"k": eng.cfg.k, "n": n, "answer_recall": hits / n,
            "recon_token_f1": sum(f1s) / n, "length_ratio": sum(ratios) / n}


def recall_markdown(results: list[dict]) -> str:
    rows = ["| K | n | answer recall | recon token F1 | length ratio |", "|---|---|---|---|---|"]
    for r in results:
        rows.append(f"| {r['k']} | {r['n']} | {r['answer_recall']:.3f} | "
                    f"{r['recon_token_f1']:.3f} | {r['length_ratio']:.3f} |")
    return "\n".join(rows)


def write_recall_report(run_dir, result: dict) -> Path:
    d = Path(run_dir) / "eval"
    d.mkdir(exist_ok=True)
    (d / "recall.json").write_text(json.dumps(result, indent=2))
    (d / "recall.md").write_text(recall_markdown([result]) + "\n")
    return d / "recall.json"
