"""Cloze recall: blank the gold answers of SQuAD dev paragraphs (first chunk only) and decode them
in cloze layout from memory. Reports hit rate (a gold answer inside the first ` | ` segment) and
teacher-forced answer loss under real / shuffled / zeroed memory."""

import json
from pathlib import Path

import torch

from engram.data.cloze import JOIN, make_gold_cloze_example
from engram.eval.ablation import CONDITIONS
from engram.eval.recall import answer_in_text

N_PARAGRAPHS = 40
CLOZE_DEV_SAMPLES = 3000  # dev examples scanned (cached) to find 40 distinct paragraphs


def select_recall_paragraphs(tokenizer, examples: list[dict], max_len: int,
                             n: int = N_PARAGRAPHS) -> list[tuple[dict, dict]]:
    """First n distinct paragraphs (in example order) whose first chunk contains a gold answer, as
    (first source example of the paragraph, gold cloze example built from every question's
    answers). Duplicate paragraphs and paragraphs without an answer in the first chunk are skipped."""
    groups: dict[tuple, tuple[dict, list[str]]] = {}
    for ex in examples:
        groups.setdefault(tuple(ex["chunks"].flatten().tolist()), (ex, []))[1].extend(ex["answers"])
    out = []
    for src, answers in groups.values():
        cloze = make_gold_cloze_example(tokenizer, src["chunks"][0], answers, max_len)
        if cloze is not None:
            out.append((src, {**cloze, "answers": sorted(set(answers))}))
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def _answer_loss(eng, cloze: list[dict], transform, batch_size: int) -> float:
    batches = [cloze[i : i + batch_size] for i in range(0, len(cloze), batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:  # a batch of 1 cannot be shuffled
        batches[-2] += batches.pop()
    return sum(float(eng.terse_loss(b, transform)) * len(b) for b in batches) / len(cloze)


@torch.no_grad()
def evaluate_cloze_recall(eng, cloze: list[dict], max_new_tokens: int = 48, batch_size: int = 8) -> dict:
    """cloze: gold cloze examples (chunks, question, target, answers). Generation goes through
    Decoder.generate; shuffled memory is another example's (roll within the paragraph list)."""
    eng.eval()
    n = len(cloze)
    if n < 2:
        raise ValueError("need at least 2 examples for shuffled memory")
    mems = [eng.encode(e["chunks"].reshape(1, -1)) for e in cloze]
    ctrl = eng._control(cloze[0]["density"], 1)
    result = {"k": eng.cfg.k, "n": n}
    for c, transform in CONDITIONS.items():
        hits = 0
        for i, e in enumerate(cloze):
            mem = {"real": mems[i], "zeroed": torch.zeros_like(mems[i]),
                   "shuffled": mems[i - 1]}[c]
            ids = eng.decoder.generate(mem, ctrl, max_new_tokens, eng.eos_id,
                                       question_ids=e["question"].reshape(1, -1))
            text = eng.tokenizer.decode(ids, skip_special_tokens=True)
            hits += answer_in_text(e["answers"], text.split(JOIN)[0])
        result[c] = {"hits": hits, "hit_rate": hits / n,
                     "answer_loss": _answer_loss(eng, cloze, transform, batch_size)}
    return result


def cloze_recall_markdown(results: list[dict]) -> str:
    rows = ["| K | n | real hits | shuffled hits | zeroed hits | real loss | shuffled loss | zeroed loss |",
            "|---|---|---|---|---|---|---|---|"]
    for r in results:
        rows.append(f"| {r['k']} | {r['n']} | " + " | ".join(
            f"{r[c]['hits']}/{r['n']}" for c in ("real", "shuffled", "zeroed")) + " | " + " | ".join(
            f"{r[c]['answer_loss']:.3f}" for c in ("real", "shuffled", "zeroed")) + " |")
    return "\n".join(rows)


def write_cloze_recall_report(run_dir, result: dict) -> Path:
    d = Path(run_dir) / "eval"
    d.mkdir(exist_ok=True)
    (d / "cloze_recall.json").write_text(json.dumps(result, indent=2))
    (d / "cloze_recall.md").write_text(cloze_recall_markdown([result]) + "\n")
    return d / "cloze_recall.json"
