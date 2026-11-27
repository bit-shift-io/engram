"""Cloze recall: blank the gold answers of SQuAD dev paragraphs (first chunk only) and decode them
in cloze layout from memory. Reports cloze recall (blank answers in the first N ` | ` segments, any order, one segment per blank),
aligned recall (segment k matches blank k) and
teacher-forced answer loss under real / shuffled / zeroed memory."""

import json
from pathlib import Path

import torch

from engram.data.cloze import JOIN, make_gold_cloze_example
from engram.eval.ablation import CONDITIONS
from engram.eval.recall import answer_in_text


def score_blanks(blanks: list[list[str]], text: str) -> tuple[int, int]:
    """(cloze recall, aligned recall) blank counts for decoded `text`: only the first len(blanks)
    ` | ` segments count; cloze recall is a max matching of blanks to distinct segments."""
    segs = text.split(JOIN)[:len(blanks)]
    ok = [[answer_in_text(b, s) for s in segs] for b in blanks]
    owner: dict[int, int] = {}

    def assign(i: int, seen: set) -> bool:
        for j, good in enumerate(ok[i]):
            if good and j not in seen:
                seen.add(j)
                if j not in owner or assign(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    recall = sum(assign(i, set()) for i in range(len(blanks)))
    return recall, sum(ok[i][i] for i in range(min(len(blanks), len(segs))))

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
            out.append((src, {**cloze, "answers": sorted(set(answers)),
                              **({"title": src["title"]} if "title" in src else {})}))
        if len(out) >= n:
            break
    return out


def other_article_pool(examples: list[dict], exclude_titles: set[str], n: int = N_PARAGRAPHS) -> list[dict]:
    """First n distinct paragraphs (in example order) whose article title is not in exclude_titles:
    lenders of shuffled memory for evaluate_cloze_recall. Dev split only, so never trained on."""
    seen: set[tuple] = set()
    out = []
    for ex in examples:
        key = tuple(ex["chunks"].flatten().tolist())
        if ex["title"] in exclude_titles or key in seen:
            continue
        seen.add(key)
        out.append(ex)
        if len(out) >= n:
            break
    return out


def cross_article_partners(titles: list[str]) -> list[int]:
    """For each paragraph, the index of a paragraph from a different article (title): the r-th
    paragraph of an article gets the r-th (cyclically) of all other articles' paragraphs, so
    partners spread out even when one article dominates. Raises if there is only one article."""
    if len(set(titles)) < 2:
        raise ValueError("shuffled memory needs paragraphs from at least 2 articles")
    rank: dict[str, int] = {}
    out = []
    for t in titles:
        others = [j for j, u in enumerate(titles) if u != t]
        out.append(others[rank.get(t, 0) % len(others)])
        rank[t] = rank.get(t, 0) + 1
    return out


@torch.no_grad()
def _answer_loss(eng, cloze: list[dict], transform, batch_size: int) -> float:
    batches = [list(range(i, min(i + batch_size, len(cloze)))) for i in range(0, len(cloze), batch_size)]
    return sum(float(eng.terse_loss([cloze[j] for j in b], transform(b))) * len(b)
               for b in batches) / len(cloze)


@torch.no_grad()
def evaluate_cloze_recall(eng, cloze: list[dict], max_new_tokens: int = 48, batch_size: int = 8,
                          shuffle_pool: list[dict] | None = None) -> dict:
    """cloze: gold cloze examples (chunks, question, target, answers). Generation goes through
    Decoder.generate; shuffled memory is a paragraph's from a different article (`title`), in generation and
    answer loss alike. shuffle_pool: extra single-chunk paragraphs (chunks, title) that may lend shuffled
    memory but are not scored — needed when the scored paragraphs span one article."""
    eng.eval()
    n = len(cloze)
    lenders = cloze + list(shuffle_pool or [])
    partner = cross_article_partners([e["title"] for e in lenders])[:n]
    used = set(partner)
    mems = [eng.encode(e["chunks"].reshape(1, -1)) for e in cloze]
    mems += [eng.encode(e["chunks"][:1].reshape(1, -1)) if j in used else None
             for j, e in enumerate(lenders[n:], start=n)]
    batch_transform = {"real": lambda b: CONDITIONS["real"], "zeroed": lambda b: CONDITIONS["zeroed"],
                       "shuffled": lambda b: lambda m: torch.cat([mems[partner[j]] for j in b]).to(m)}
    ctrl = eng._control(cloze[0]["density"], 1)
    total = sum(len(e["blanks"]) for e in cloze)
    result = {"k": eng.cfg.k, "n": n}
    for c in CONDITIONS:
        recalled = aligned = 0
        for i, e in enumerate(cloze):
            mem = {"real": mems[i], "zeroed": torch.zeros_like(mems[i]),
                   "shuffled": mems[partner[i]]}[c]
            ids = eng.decoder.generate(mem, ctrl, max_new_tokens, eng.eos_id,
                                       question_ids=e["question"].reshape(1, -1))
            text = eng.tokenizer.decode(ids, skip_special_tokens=True)
            r_, a_ = score_blanks(e["blanks"], text)
            recalled, aligned = recalled + r_, aligned + a_
        result[c] = {"blanks": total, "cloze_recall": recalled / total, "aligned_recall": aligned / total,
                     "answer_loss": _answer_loss(eng, cloze, batch_transform[c], batch_size)}
    return result


def cloze_recall_markdown(results: list[dict]) -> str:
    conds = ("real", "shuffled", "zeroed")
    rows = ["| K | n | blanks | " + " | ".join(f"{c} cloze recall | {c} aligned recall | {c} loss" for c in conds) + " |",
            "|---|---|---|" + "---|" * (3 * len(conds))]
    for r in results:
        rows.append(f"| {r['k']} | {r['n']} | {r['real']['blanks']} | " + " | ".join(
            f"{r[c]['cloze_recall']:.3f} | {r[c]['aligned_recall']:.3f} | {r[c]['answer_loss']:.3f}"
            for c in conds) + " |")
    return "\n".join(rows)


def write_cloze_recall_report(run_dir, result: dict, eval_dir: str = "eval") -> Path:
    d = Path(run_dir) / eval_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "cloze_recall.json").write_text(json.dumps(result, indent=2))
    (d / "cloze_recall.md").write_text(cloze_recall_markdown([result]) + "\n")
    return d / "cloze_recall.json"
