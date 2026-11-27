"""SQuAD QA eval: EM/F1 of `<answer>` generation with real / shuffled / zeroed memory, plus the
frozen base model zero-shot with paragraph + question in context."""

import json
import re
import string
from collections import Counter
from pathlib import Path

import torch

from engram.eval.density import strip_trailing_eos
from engram.model.base import generate

CONDITIONS = ("real", "shuffled", "zeroed")


def find_word_bounded(needle: str, text: str) -> int:
    """Index of the first occurrence of `needle` in `text` not preceded or followed by an
    alphanumeric character, else -1. Look-arounds (not `\\b`) so edge punctuation still matches."""
    if not needle:
        return -1
    m = re.search(r"(?<![^\W_])" + re.escape(needle) + r"(?![^\W_])", text)
    return m.start() if m else -1


def normalize(s: str) -> str:
    """SQuAD v1.1 normalisation: lowercase, drop punctuation and articles, squeeze whitespace."""
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    return float(any(normalize(pred) == normalize(g) for g in golds))


def _f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_score(pred: str, golds: list[str]) -> float:
    return max(_f1(pred, g) for g in golds)


def _answer_text(eng, ids: list[int]) -> str:
    """Decode up to EOS (already cut) and the first newline."""
    text = eng.tokenizer.decode(strip_trailing_eos(ids, eng.eos_id), skip_special_tokens=True)
    return text.strip().split("\n")[0].strip()


def _score(preds: list[str], examples: list[dict]) -> dict:
    n = len(examples)
    return {"em": sum(exact_match(p, e["answers"]) for p, e in zip(preds, examples)) / n,
            "f1": sum(f1_score(p, e["answers"]) for p, e in zip(preds, examples)) / n}


def text_qa_prompt(tokenizer, ex: dict, eos_id: int) -> list[int]:
    """Plain-text QA prompt: `paragraph\nQuestion: q\nAnswer:` (zero-shot and text-QA control)."""
    ctx = strip_trailing_eos(ex["chunks"].flatten().tolist(), eos_id)
    nl = lambda s: tokenizer(s, add_special_tokens=False).input_ids
    return ctx + nl("\nQuestion:") + ex["question"].tolist() + nl("\nAnswer:")


def text_qa_example(tokenizer, ex: dict, eos_id: int) -> tuple[list[int], list[int]]:
    """(input_ids, labels) for the text-QA control: prompt + answer + EOS; prompt labels = -100."""
    prompt = text_qa_prompt(tokenizer, ex, eos_id)
    target = ex["target"].tolist()
    return prompt + target, [-100] * len(prompt) + target


@torch.no_grad()
def zero_shot_answer(eng, ex: dict, max_new_tokens: int) -> str:
    """Base model (adapters off), prompt: paragraph, question, 'Answer:'."""
    with eng.lm.disable_adapter():
        return text_qa_answer(eng.lm, eng.tokenizer, ex, eng.cfg.device, max_new_tokens)


@torch.no_grad()
def text_qa_answer(lm, tokenizer, ex: dict, device: str, max_new_tokens: int) -> str:
    """Greedy answer to the plain-text QA prompt, cut at EOS and the first newline."""
    eos = tokenizer.eos_token_id
    out = generate(lm, torch.tensor([text_qa_prompt(tokenizer, ex, eos)]), device, max_new_tokens)
    ids = out[0].tolist()
    for i, t in enumerate(ids):  # stop at EOS
        if t == eos:
            ids = ids[:i]
            break
    return tokenizer.decode(ids, skip_special_tokens=True).strip().split("\n")[0].strip()


@torch.no_grad()
def evaluate_text_qa(lm, tokenizer, examples: list[dict], device: str,
                     max_new_tokens: int = 32) -> dict:
    """EM/F1 of the text-QA control (same prompt layout and scorer as base zero-shot)."""
    lm.eval()
    preds = [text_qa_answer(lm, tokenizer, e, device, max_new_tokens) for e in examples]
    return {"n": len(examples), **_score(preds, examples)}


@torch.no_grad()
def evaluate_qa(eng, examples: list[dict], max_new_tokens: int = 32, zero_shot: bool = True,
                seed: int | None = None) -> dict:
    """examples need chunks, question, answers. Shuffled = another example's memory (random
    derangement by seed). Generation goes through Decoder.generate (memory hooks on)."""
    eng.eval()
    n = len(examples)
    if n < 2:
        raise ValueError("need at least 2 examples for shuffled memory")
    seed = eng.cfg.qa_eval_seed if seed is None else seed
    gen = torch.Generator().manual_seed(seed)
    shift_perm = torch.randperm(n, generator=gen).tolist()
    # derangement: rotate along a random order so no example keeps its own memory
    other = {shift_perm[i]: shift_perm[(i + 1) % n] for i in range(n)}
    mems = [eng.encode(e["chunks"].reshape(1, -1)) for e in examples]
    ctrl = eng._control("answer", 1)
    preds = {c: [] for c in CONDITIONS}
    for i, e in enumerate(examples):
        q = e["question"].reshape(1, -1)
        for c, m in (("real", mems[i]), ("shuffled", mems[other[i]]),
                     ("zeroed", torch.zeros_like(mems[i]))):
            ids = eng.decoder.generate(m, ctrl, max_new_tokens, eng.eos_id, question_ids=q)
            preds[c].append(_answer_text(eng, ids))
    result = {"k": eng.cfg.k, "n": n, "seed": seed, **{c: _score(preds[c], examples) for c in CONDITIONS}}
    if zero_shot:
        result["base_zero_shot"] = _score([zero_shot_answer(eng, e, max_new_tokens) for e in examples],
                                          examples)
    return result


def _ratio(r: dict, control: dict | None) -> float | None:
    if not control or not control["f1"]:
        return None
    return r["real"]["f1"] / control["f1"]


def qa_markdown(results: list[dict], control: dict | None = None) -> str:
    """With a text-QA control result, adds control EM/F1 and real-memory F1 / control F1."""
    head = ("| K | n | real EM | real F1 | shuffled EM | shuffled F1 | zeroed EM | zeroed F1 | "
            "base zero-shot EM | base zero-shot F1 |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    if control:
        head += " text-QA control EM | text-QA control F1 | memory ÷ control F1 |"
        sep += "---|---|---|"
    rows = [head, sep]
    for r in results:
        cells = [f"{r[c][m]:.3f}" for c in CONDITIONS for m in ("em", "f1")]
        b = r.get("base_zero_shot")
        cells += [f"{b['em']:.3f}", f"{b['f1']:.3f}"] if b else ["n/a", "n/a"]
        if control:
            ratio = _ratio(r, control)
            cells += [f"{control['em']:.3f}", f"{control['f1']:.3f}",
                      f"{ratio:.3f}" if ratio is not None else "n/a"]
        rows.append(f"| {r['k']} | {r['n']} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_qa_report(run_dir, result: dict, control: dict | None = None) -> Path:
    d = Path(run_dir) / "eval"
    d.mkdir(exist_ok=True)
    if control:
        result = {**result, "text_qa_control": control, "memory_over_control_f1": _ratio(result, control)}
    (d / "qa.json").write_text(json.dumps(result, indent=2))
    (d / "qa.md").write_text(qa_markdown([result], control) + "\n")
    return d / "qa.json"


def _text_qa_batch(tokenizer, examples: list[dict], eos_id: int):
    pairs = [text_qa_example(tokenizer, e, eos_id) for e in examples]
    n = max(len(ids) for ids, _ in pairs)
    ids = torch.full((len(pairs), n), eos_id)
    labels = torch.full((len(pairs), n), -100)
    mask = torch.zeros((len(pairs), n), dtype=torch.long)
    for i, (x, y) in enumerate(pairs):  # right-pad
        ids[i, : len(x)], labels[i, : len(y)], mask[i, : len(x)] = torch.tensor(x), torch.tensor(y), 1
    return ids, labels, mask


def train_text_qa(model, tokenizer, cfg, examples: list[dict], run_dir) -> tuple:
    """Text-QA control: one LoRA (cfg.lora_r/alpha/targets) on the frozen base, trained for
    cfg.text_qa_steps optimizer steps at cfg.lr (warmup + cosine, clipped). Saves the LoRA params
    to run_dir/ckpt-<steps>.pt. Returns (peft model, per-step train losses)."""
    from dataclasses import replace

    from peft import LoraConfig, get_peft_model

    from engram.train.trainer import lr_at

    lm = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
                                          target_modules=list(cfg.lora_targets)))
    params = [p for p in lm.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    sched = replace(cfg, steps=cfg.text_qa_steps)
    gen = torch.Generator().manual_seed(cfg.seed)
    eos = tokenizer.eos_token_id
    lm.train()
    losses = []
    for step in range(cfg.text_qa_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, sched)
        opt.zero_grad()
        total = 0.0
        for _ in range(cfg.grad_accum):
            idx = torch.randint(len(examples), (cfg.batch_size,), generator=gen).tolist()
            ids, labels, mask = _text_qa_batch(tokenizer, [examples[i] for i in idx], eos)
            dev = cfg.device
            loss = lm(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                      labels=labels.to(dev)).loss / cfg.grad_accum
            loss.backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        opt.step()
        losses.append(total)
    lm.eval()
    state = {n: p.detach().clone() for n, p in lm.named_parameters() if p.requires_grad}
    torch.save(state, Path(run_dir) / f"ckpt-{cfg.text_qa_steps}.pt")
    return lm, losses
