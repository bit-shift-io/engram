from engram.data.cloze import find_fact_spans


def spans_text(text):
    return [text[s:e] for s, e in find_fact_spans(text)]


def test_finds_numbers_dates_and_names_not_sentence_initial():
    text = "The band played in March 1999 at Notre Dame. They sold 42 records."
    assert spans_text(text) == ["March 1999", "Notre Dame", "42"]


def test_wikitext_number_separators_stay_one_span():
    assert spans_text("it cost 1 @,@ 000 dollars") == ["1 @,@ 000"]


import math

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

# 64 words (tiny model vocab) incl. fact words and the cloze prompt pieces
FACT_WORDS = ["<eos>", "<unk>", "the", "band", "played", "in", "March", "1999", "at", "Notre",
              "Dame", ".", "They", "sold", "42", "records", "___", "Answers:", "|"]
WORDS = FACT_WORDS + [f"w{i}" for i in range(64 - len(FACT_WORDS))]


def fact_tokenizer():
    tk = Tokenizer(WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tk.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>", unk_token="<unk>")


def ids(tok, text):
    return torch.tensor(tok(text, add_special_tokens=False).input_ids)


def test_cloze_example_blanks_facts_and_targets_them_in_order():
    from engram.data.cloze import make_cloze_example
    tok = fact_tokenizer()
    chunk = ids(tok, "the band played in March 1999 at Notre Dame . They sold 42 records")
    ex = make_cloze_example(tok, chunk, max_len=64)
    prompt = tok.decode(ex["question"])
    assert not any(w in prompt for w in ["March", "1999", "Notre", "Dame", "42"])
    assert prompt.count("___") == 3 and prompt.endswith("Answers:")
    assert tok.decode(ex["target"][:-1]) == "March 1999 | Notre Dame | 42"
    assert ex["target"][-1] == tok.eos_token_id
    assert torch.equal(ex["chunks"], chunk[None])


def test_cloze_example_none_without_facts_and_truncates_to_budget():
    from engram.data.cloze import make_cloze_example
    tok = fact_tokenizer()
    assert make_cloze_example(tok, ids(tok, "the band played in w3 w4"), max_len=64) is None
    chunk = ids(tok, "the band played in March 1999 at Notre Dame . They sold 42 records")
    ex = make_cloze_example(tok, chunk, max_len=10)
    assert len(ex["question"]) + len(ex["target"]) <= 10 and len(ex["target"]) >= 1


def test_cloze_sampler_share_matches_ratio_and_zero_ratio_is_unchanged():
    from engram.data.cloze import ClozeSampler
    from engram.data.mixed import MixedSampler
    chunks = torch.arange(80).reshape(10, 8)

    def draws(ratio, n=2000):
        g = torch.Generator().manual_seed(0)
        s = ClozeSampler(MixedSampler(chunks, None, 1.0, 2, g), ratio, g)
        return [s.next() for _ in range(n)]

    share = sum(d == "cloze" for d, _ in draws(0.3)) / 2000
    assert abs(share - 0.3) < 0.04
    g = torch.Generator().manual_seed(0)
    plain = MixedSampler(chunks, None, 1.0, 2, g)
    for d, b in draws(0.0, 50):
        pd, pb = plain.next()
        assert d == pd == "full" and torch.equal(b, pb)


def test_cloze_training_smoke_logs_real_shuffled_zeroed_val_loss(tmp_path):
    import json

    from conftest import make_tiny_model
    from engram.config import Config
    from engram.model.engram import Engram
    from engram.train.trainer import train
    cfg = Config(chunk_size=8, k=4, steps=4, batch_size=2, grad_accum=2, warmup_steps=1,
                 val_interval=4, ckpt_interval=4, run_dir=str(tmp_path), lr=1e-2,
                 cloze_ratio=0.5, decoder_lora=False, decoder_memory_adapter_rank=2)
    tok = fact_tokenizer()
    eng = Engram(make_tiny_model(), tok, cfg)
    fact = ids(tok, "band played in March 1999 at Notre Dame")
    plain = ids(tok, "the band played in w3 w4 w5 w6")
    train_chunks = torch.stack([fact, plain] * 4)
    val_chunks = torch.stack([fact, plain, fact, fact])
    run = train(eng, cfg, train_chunks, val_chunks)
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert sum(r.get("n_cloze", 0) for r in rows) > 0
    assert all(math.isfinite(r["train_loss"]) for r in rows if "train_loss" in r)
    val = [r for r in rows if "val_loss" in r][-1]
    for c in ("real", "shuffled", "zeroed"):
        assert math.isfinite(val[f"cloze_val_loss_{c}"])


def qa_ex(tok, text, answers, chunk=16):
    ids_ = tok(text, add_special_tokens=False).input_ids
    ids_ += [tok.eos_token_id] * (-len(ids_) % chunk)
    return {"chunks": torch.tensor(ids_).reshape(-1, chunk), "answers": answers}


def test_gold_cloze_blanks_every_answer_in_text_order_without_eos():
    from engram.data.cloze import squad_cloze_examples
    tok = fact_tokenizer()
    text = "the band played in March 1999 at Notre Dame . They sold 42 records"
    exs = [qa_ex(tok, text, ["Notre Dame"]), qa_ex(tok, text, ["March 1999", "42"])]
    out = squad_cloze_examples(tok, exs, max_len=64)
    assert len(out) == 1  # paragraph repeated per question -> one example
    prompt = tok.decode(out[0]["question"])
    assert prompt.count("___") == 3 and "<eos>" not in prompt and "Dame" not in prompt
    assert tok.decode(out[0]["target"][:-1]) == "March 1999 | Notre Dame | 42"


def test_gold_cloze_drops_answers_outside_first_chunk_and_skips_empty_paragraphs():
    from engram.data.cloze import squad_cloze_examples
    tok = fact_tokenizer()
    text = "the band played in March 1999 at Notre Dame . They sold 42 records"  # 15 tokens
    long = text + " the band played in w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 Dame"
    out = squad_cloze_examples(tok, [qa_ex(tok, long, ["42", "w16 Dame", "w20"])], max_len=64)
    assert tok.decode(out[0]["target"][:-1]) == "42"
    assert squad_cloze_examples(tok, [qa_ex(tok, text, ["w20"])], max_len=64) == []


def _dev(tok):
    text = "the band played in March 1999 at Notre Dame . They sold 42 records"
    return [qa_ex(tok, text, ["42"]), qa_ex(tok, text, ["March 1999"]),  # same paragraph twice
            qa_ex(tok, "the band played in w3 w4", ["w9"]),  # answer not in chunk
            qa_ex(tok, text.replace("42", "1999"), ["Notre Dame"]),
            qa_ex(tok, text.replace("42", "March"), ["Dame"])]


def test_select_recall_paragraphs_skips_duplicates_and_missing_answers_and_stops_at_n():
    from engram.eval.cloze_recall import select_recall_paragraphs
    tok = fact_tokenizer()
    dev = _dev(tok)
    sel = select_recall_paragraphs(tok, dev, max_len=64)
    assert [s for s, _ in sel] == [dev[0], dev[3], dev[4]]
    assert sorted(sel[0][1]["answers"]) == ["42", "March 1999"]  # answers of both questions
    assert [s for s, _ in select_recall_paragraphs(tok, dev, max_len=64, n=2)] == [dev[0], dev[3]]


def test_trainer_validation_excludes_exactly_the_recall_eval_paragraphs():
    from engram.eval.cloze_recall import select_recall_paragraphs
    from engram.train.trainer import squad_cloze_val_examples
    tok = fact_tokenizer()
    dev = _dev(tok)
    # the eval takes n=40 distinct paragraphs; with more than 40 in dev, the rest stay for validation
    extra = [qa_ex(tok, f"the band played in March 1999 at w{i} Dame", ["Dame"]) for i in range(20)]
    extra += [qa_ex(tok, f"they sold w{i} records at Notre w{j} Dame .", ["Notre"])
              for i in range(5) for j in range(5, 9)]
    dev = dev + extra
    picked = {tuple(s["chunks"].flatten().tolist()) for s, _ in select_recall_paragraphs(tok, dev, 64)}
    val = squad_cloze_val_examples(tok, dev, 64)
    val_keys = {tuple(v["chunks"].flatten().tolist()) for v in val}
    all_keys = {tuple(d["chunks"][0].tolist()) for d in dev}
    assert len(picked) == 40 and val  # full recall set held out, something left to validate on
    assert not (picked & val_keys) and len(val) == len(all_keys - picked) - 1  # minus the no-answer paragraph


class StubEng:
    """Decoder stub: generation replays scripted text per memory id; loss is scripted per condition."""
    class cfg:
        k = 4
        device = "cpu"
    eos_id = 0

    def __init__(self, tok, outputs):
        self.tokenizer, self.outputs, self.decoder = tok, outputs, self
        self.seen = []

    def eval(self):
        pass

    def encode(self, ids):  # memory value = first token id
        return ids[:, :1].float().unsqueeze(-1)

    def _control(self, density, b):
        return torch.zeros(b, 1, dtype=torch.long)

    def generate(self, mem, ctrl, n, eos, question_ids=None):
        return self.tokenizer(self.outputs.get(round(float(mem.flatten()[0])), "w1"), add_special_tokens=False).input_ids

    def terse_loss(self, examples, memory_transform=None):
        mem = torch.tensor([[float(e["chunks"][0, 0])] for e in examples]).unsqueeze(-1)
        m = mem if memory_transform is None else memory_transform(mem)
        return (m == 0).float().mean() * 4 + (m != mem).float().mean() * 2 + 1  # real 1, shuffled 3, zeroed 5


def test_evaluate_cloze_recall_hit_rate_and_loss_per_condition():
    from engram.eval.cloze_recall import evaluate_cloze_recall
    tok = fact_tokenizer()
    exs = [{"chunks": ids(tok, t)[None], "question": ids(tok, "the ___ Answers:"), "density": "full",
            "answers": [a]} for t, a in [("March w3", "42"), ("Notre w4", "Dame"), ("42 w5", "records"),
                                         ("Dame w6", "1999")]]
    # memory id -> generated text; real memory answers right in first segment for 2 of 4
    outs = {int(exs[0]["chunks"][0, 0]): "42 | w1", int(exs[1]["chunks"][0, 0]): "Dame | w1",
            int(exs[2]["chunks"][0, 0]): "w1 | records", int(exs[3]["chunks"][0, 0]): "w1"}
    r = evaluate_cloze_recall(StubEng(tok, outs), exs, batch_size=4)
    assert r["real"]["hits"] == 2 and r["real"]["hit_rate"] == 0.5 and r["n"] == 4
    assert r["shuffled"]["hits"] == 0  # shifted memory yields the neighbour's text
    assert r["zeroed"]["hits"] == 0
    assert r["real"]["answer_loss"] < r["shuffled"]["answer_loss"] < r["zeroed"]["answer_loss"]
