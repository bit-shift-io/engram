"""Cloze task: the encoder sees a chunk; the decoder sees the chunk with its fact spans blanked
(`___`) plus a separator, and must produce the spans in order (` | `-joined) from memory."""

import re

import torch

MONTHS = {"January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"}
BLANK, SEP, JOIN = "___", "\nAnswers:", " | "

# digit runs incl. WikiText's " @,@ " / " @.@ " number separators, or a word
_WORD = re.compile(r"\b\d+(?:(?: @[,.]@ |[.,])\d+)*|[A-Za-z][A-Za-z0-9'\-]*")
_SENT_END = re.compile(r"[.!?]\s*$")


def _is_fact(text: str, m: re.Match) -> bool:
    w = m.group()
    if w[0].isdigit() or w in MONTHS:
        return True
    if not w[0].isupper():
        return False
    before = text[: m.start()].rstrip()
    return bool(before) and not _SENT_END.search(before) and not before.endswith("=")


def find_fact_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of facts: numbers/years, month names, capitalised words that do not start
    a sentence. Adjacent facts separated only by spaces merge ("March 1999", "Notre Dame")."""
    spans: list[list[int]] = []
    for m in _WORD.finditer(text):
        if not _is_fact(text, m):
            continue
        if spans and text[spans[-1][1] : m.start()].strip(" ") == "":
            spans[-1][1] = m.end()
        else:
            spans.append([m.start(), m.end()])
    return [(s, e) for s, e in spans]


def make_cloze_example(tokenizer, chunk_ids: torch.Tensor, max_len: int) -> dict | None:
    """{chunks [1, C], question (blanked chunk + separator), target (spans + EOS), density} in the
    `<answer>` example layout, so the question is attended but not supervised; None if the chunk
    has no fact spans. Spans are found on decoded text and re-tokenised; question + target are
    truncated to max_len (target to at most half, question from the start so no fact leaks)."""
    text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
    spans = find_fact_spans(text)
    if not spans:
        return None
    return _layout(tokenizer, chunk_ids, text, spans, max_len)


def _layout(tokenizer, chunk_ids, text, spans, max_len):
    parts, prev = [], 0
    for s, e in spans:
        parts += [text[prev:s], BLANK]
        prev = e
    prompt = "".join(parts) + text[prev:] + SEP
    answer = JOIN.join(text[s:e] for s, e in spans)
    q = tokenizer(prompt, add_special_tokens=False).input_ids
    a = tokenizer(answer, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    a = a[: max(1, max_len // 2) if len(q) + len(a) > max_len else len(a)]
    q = q[: max_len - len(a)]
    return {"chunks": chunk_ids.reshape(1, -1), "question": torch.tensor(q, dtype=torch.long),
            "target": torch.tensor(a, dtype=torch.long), "density": "full"}


def paragraph_answers(examples: list[dict]) -> list[tuple[torch.Tensor, list[str]]]:
    """(first chunk with EOS stripped, gold answers of every question) per distinct paragraph;
    SQuAD examples repeat a paragraph once per question."""
    groups: dict[tuple, tuple[torch.Tensor, list[str]]] = {}
    for ex in examples:
        key = tuple(ex["chunks"].flatten().tolist())
        groups.setdefault(key, (ex["chunks"][0], []))[1].extend(ex["answers"])
    return list(groups.values())


def make_gold_cloze_example(tokenizer, chunk_ids: torch.Tensor, answers: list[str],
                            max_len: int) -> dict | None:
    """Cloze example whose blanks are SQuAD gold answers found in the (EOS-stripped) chunk, in text
    order; answers not in the chunk are dropped, None if none remain. Same layout as
    `make_cloze_example`."""
    ids = chunk_ids.tolist()
    while ids and ids[-1] == tokenizer.eos_token_id:
        ids.pop()
    text = tokenizer.decode(ids, skip_special_tokens=True)
    found = sorted({(i, i + len(a)) for a in set(answers) if a.strip()
                    for i in [text.find(a)] if i >= 0})
    spans: list[tuple[int, int]] = []
    for s, e in found:
        if spans and s < spans[-1][1]:
            spans[-1] = (spans[-1][0], max(e, spans[-1][1]))
        else:
            spans.append((s, e))
    if not spans:
        return None
    return _layout(tokenizer, chunk_ids, text, spans, max_len)  # memory keeps the full chunk


def squad_cloze_examples(tokenizer, examples: list[dict], max_len: int,
                         exclude: list[dict] | None = None) -> list[dict]:
    """Gold-answer cloze examples, one per distinct paragraph of SQuAD `examples` (paragraphs of
    `exclude` are skipped)."""
    skip = {tuple(e["chunks"].flatten().tolist()) for e in exclude or []}
    out = []
    for chunk, answers in paragraph_answers([e for e in examples
                                             if tuple(e["chunks"].flatten().tolist()) not in skip]):
        ex = make_gold_cloze_example(tokenizer, chunk, answers, max_len)
        if ex is not None:
            out.append(ex)
    return out


class ClozeSampler:
    """Wraps a MixedSampler: each <full> chunk batch becomes ("cloze", same chunks) with
    probability `ratio`; of the remaining <full> batches, a share `squad_ratio` becomes
("squad_cloze", list of B gold-answer cloze examples). Ratio 0 draws no extra randomness, so
batches are unchanged."""

    def __init__(self, sampler, ratio: float, generator: torch.Generator,
                 squad_cloze: list[dict] | None = None, squad_ratio: float = 0.0,
                 batch_size: int = 1):
        self.sampler, self.ratio, self.gen = sampler, ratio, generator
        self.squad, self.squad_ratio, self.batch_size = squad_cloze or [], squad_ratio, batch_size
        if squad_ratio > 0 and not self.squad:
            raise ValueError("squad_cloze_ratio > 0 needs squad cloze examples")

    def next(self) -> tuple[str, object]:
        density, batch = self.sampler.next()
        if density == "full" and self.ratio and float(torch.rand((), generator=self.gen)) < self.ratio:
            return "cloze", batch
        if (density == "full" and self.squad_ratio
                and float(torch.rand((), generator=self.gen)) < self.squad_ratio):
            idx = torch.randint(len(self.squad), (self.batch_size,), generator=self.gen)
            return "squad_cloze", [self.squad[i] for i in idx.tolist()]
        return density, batch
