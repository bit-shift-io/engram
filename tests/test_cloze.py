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
