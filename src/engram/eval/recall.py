"""Answer recall: does a `<full>` reconstruction of a SQuAD paragraph still contain the gold
answer span? Also reconstruction token F1 and length ratio (reconstruction / paragraph tokens)."""

import json
from pathlib import Path

import torch

from engram.eval.qa import f1_score, find_word_bounded, normalize


def answer_in_text(answers: list[str], text: str) -> bool:
    """True if any non-empty normalised gold answer occurs on word boundaries in the normalised text."""
    t = normalize(text)
    return any(find_word_bounded(a, t) >= 0 for a in map(normalize, answers))


@torch.no_grad()
def evaluate_recall(eng, examples: list[dict]) -> dict:
    """examples need chunks and answers. Generation goes through Engram.generate (memory hooks on)."""
    eng.eval()
    tok = eng.tokenizer
    hits, zeroed_hits, f1s, ratios = 0, 0, [], []
    for ex in examples:
        ids = ex["chunks"].flatten().tolist()
        ctx = tok.decode(ids, skip_special_tokens=True)
        rec = eng.generate(ctx, "full")
        hits += answer_in_text(ex["answers"], rec)
        # control: how often the answer appears with no memory at all (guessing from the decoder alone)
        zeroed_hits += answer_in_text(ex["answers"], eng.generate(ctx, "full", memory_transform=torch.zeros_like))
        f1s.append(f1_score(rec, [ctx]))
        n_ctx = len(tok(ctx, add_special_tokens=False).input_ids)
        n_rec = len(tok(rec, add_special_tokens=False).input_ids)
        ratios.append(n_rec / max(n_ctx, 1))
    n = len(examples)
    return {"k": eng.cfg.k, "n": n, "answer_recall": hits / n, "zeroed_answer_recall": zeroed_hits / n,
            "recon_token_f1": sum(f1s) / n, "length_ratio": sum(ratios) / n}


def recall_markdown(results: list[dict]) -> str:
    rows = ["| K | n | answer recall | zeroed answer recall | recon token F1 | length ratio |",
            "|---|---|---|---|---|---|"]
    for r in results:
        z = r.get("zeroed_answer_recall")
        rows.append(f"| {r['k']} | {r['n']} | {r['answer_recall']:.3f} | {'n/a' if z is None else f'{z:.3f}'} | "
                    f"{r['recon_token_f1']:.3f} | {r['length_ratio']:.3f} |")
    return "\n".join(rows)


def write_recall_report(run_dir, result: dict, eval_dir: str = "eval") -> Path:
    d = Path(run_dir) / eval_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "recall.json").write_text(json.dumps(result, indent=2))
    (d / "recall.md").write_text(recall_markdown([result]) + "\n")
    return d / "recall.json"
