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


class ClozeSampler:
    """Wraps a MixedSampler: each <full> chunk batch becomes ("cloze", same chunks) with
    probability `ratio`. ratio 0 draws no extra randomness, so batches are unchanged."""

    def __init__(self, sampler, ratio: float, generator: torch.Generator):
        self.sampler, self.ratio, self.gen = sampler, ratio, generator

    def next(self) -> tuple[str, object]:
        density, batch = self.sampler.next()
        if density == "full" and self.ratio and float(torch.rand((), generator=self.gen)) < self.ratio:
            return "cloze", batch
        return density, batch
