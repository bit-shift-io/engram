import torch

from engram.data.compression import make_example
from conftest import make_tiny_tokenizer


def test_article_becomes_at_most_four_eos_padded_chunks_with_terse_target():
    tok = make_tiny_tokenizer()
    article = " ".join(f"w{i % 60}" for i in range(50))  # 50 tokens -> 7 chunks of 8 untruncated
    ex = make_example(tok, article, "w1 w2 w3", chunk_size=8, max_chunks=4)
    assert ex["chunks"].shape == (4, 8)
    assert ex["chunks"].flatten().tolist() == tok(article).input_ids[:32]
    assert ex["target"].tolist() == tok("w1 w2 w3").input_ids + [tok.eos_token_id]
    assert ex["density"] == "terse"

    short = make_example(tok, "w5 w6 w7", "w1", chunk_size=8, max_chunks=4)
    assert short["chunks"].shape == (1, 8)
    assert short["chunks"][0].tolist() == tok("w5 w6 w7").input_ids + [tok.eos_token_id] * 5


def test_mixed_sampler_ratio_matches_config():
    from engram.data.mixed import MixedSampler

    full = torch.randint(2, 64, (20, 8))
    terse = [make_example(make_tiny_tokenizer(), "w1 w2", "w3", 8, 4) for _ in range(10)]
    s = MixedSampler(full, terse, full_ratio=0.7, batch_size=3,
                     generator=torch.Generator().manual_seed(0))
    kinds = []
    for _ in range(2000):
        density, batch = s.next()
        kinds.append(density)
        if density == "full":
            assert batch.shape == (3, 8)
        else:
            assert len(batch) == 3 and all(e["density"] == "terse" for e in batch)
    assert abs(kinds.count("full") / len(kinds) - 0.7) < 0.03


def test_terse_loss_with_variable_chunk_counts_ignores_memory_padding(tiny_engram):
    eng = tiny_engram(chunk_size=8, k=4).eval()
    tok = eng.tokenizer
    a = make_example(tok, "w3 w4 w5", "w1 w2", 8, 4)  # 1 chunk
    b = make_example(tok, " ".join(f"w{i}" for i in range(20)), "w6 w7", 8, 4)  # 3 chunks
    with torch.no_grad():
        la, lb = float(eng.terse_loss([a])), float(eng.terse_loss([b]))
        both = float(eng.terse_loss([a, b]))
    assert abs(both - (la + lb) / 2) < 1e-4


def test_full_example_reconstructs_the_article_without_padding():
    from engram.data.compression import full_example

    tok = make_tiny_tokenizer()
    eos = tok.eos_token_id
    ex = make_example(tok, " ".join(f"w{i}" for i in range(20)), "w1", chunk_size=8, max_chunks=4)
    fx = full_example(ex, eos)
    assert fx["density"] == "full"
    assert torch.equal(fx["chunks"], ex["chunks"])  # same multi-chunk memory input
    assert fx["target"].tolist() == tok(" ".join(f"w{i}" for i in range(20))).input_ids + [eos]


def test_single_chunk_full_example_matches_plain_reconstruction_loss(tiny_engram):
    from engram.data.compression import full_example

    eng = tiny_engram(chunk_size=8, k=4).eval()
    tok = eng.tokenizer
    ex = make_example(tok, " ".join(f"w{i}" for i in range(8)), "w1", 8, 4)  # exactly one chunk
    with torch.no_grad():
        multi = float(eng.terse_loss([full_example(ex, eng.eos_id)]))
        plain = float(eng.loss(ex["chunks"]))
    assert abs(multi - plain) < 1e-4


def test_mixed_sampler_draws_multi_chunk_full_batches():
    from engram.data.compression import full_example
    from engram.data.mixed import MixedSampler

    tok = make_tiny_tokenizer()
    full = torch.randint(2, 64, (20, 8))
    terse = [make_example(tok, "w1 w2", "w3", 8, 4) for _ in range(10)]
    multi = [full_example(e, tok.eos_token_id) for e in terse]
    s = MixedSampler(full, terse, full_ratio=0.5, batch_size=3,
                     generator=torch.Generator().manual_seed(0), multi_full=multi, multi_ratio=0.5)
    kinds = [s.next() for _ in range(2000)]
    names = [k for k, _ in kinds]
    for k, b in kinds:
        if k == "multi":
            assert len(b) == 3 and all(e["density"] == "full" for e in b)
    assert abs(names.count("terse") / 2000 - 0.5) < 0.04
    assert abs(names.count("multi") / 2000 - 0.25) < 0.04
    assert abs(names.count("full") / 2000 - 0.25) < 0.04
