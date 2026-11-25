import torch

from conftest import make_tiny_tokenizer


def test_paragraph_becomes_eos_padded_chunks_with_question_and_answer_target():
    from engram.data.qa import make_qa_example

    tok = make_tiny_tokenizer()
    ctx = " ".join(f"w{i}" for i in range(20))  # 20 tokens -> 3 chunks of 8
    ex = make_qa_example(tok, ctx, "w40 w41", "w5 w6", chunk_size=8, max_chunks=4)
    assert ex["chunks"].shape == (3, 8)
    assert ex["chunks"].flatten().tolist() == tok(ctx).input_ids + [tok.eos_token_id] * 4
    assert ex["question"].tolist() == tok("w40 w41").input_ids
    assert ex["target"].tolist() == tok("w5 w6").input_ids + [tok.eos_token_id]
    assert ex["density"] == "answer"


def test_paragraph_over_max_chunks_is_skipped():
    from engram.data.qa import make_qa_example

    tok = make_tiny_tokenizer()
    ctx = " ".join(f"w{i % 60}" for i in range(33))  # 5 chunks of 8
    assert make_qa_example(tok, ctx, "w1", "w2", chunk_size=8, max_chunks=4) is None


def test_answer_loss_masks_question_positions(tiny_engram):
    from engram.data.qa import make_qa_example

    eng = tiny_engram(chunk_size=8, k=4, control_vectors=True).eval()
    tok = eng.tokenizer
    seen = {}
    orig = eng.lm.forward

    def spy(*a, **kw):
        seen["labels"] = kw.get("labels")
        return orig(*a, **kw)

    eng.lm.forward = spy
    a = make_qa_example(tok, "w3 w4 w5", "w10 w11 w12", "w4", 8, 4)
    b = make_qa_example(tok, " ".join(f"w{i}" for i in range(20)), "w13", "w6 w7", 8, 4)
    with torch.no_grad():
        loss = eng.terse_loss([a, b])
    assert torch.isfinite(loss)
    labels = seen["labels"]
    m = 3 * 4 + 1  # padded memory (3 chunks x k) + <answer>
    # a: question (3) ignored, answer + EOS supervised, padding ignored
    assert labels[0, m:].tolist() == [-100] * 3 + a["target"].tolist() + [-100] * 0
    # b: question (1) ignored, answer + EOS, then padding
    assert labels[1, m:].tolist() == [-100] + b["target"].tolist() + [-100] * 1


def test_mixed_sampler_answer_share():
    from engram.data.compression import make_example
    from engram.data.mixed import MixedSampler
    from engram.data.qa import make_qa_example

    tok = make_tiny_tokenizer()
    full = torch.randint(2, 64, (20, 8))
    terse = [make_example(tok, "w1 w2", "w3", 8, 4) for _ in range(10)]
    qa = [make_qa_example(tok, "w1 w2", "w3", "w4", 8, 4) for _ in range(10)]
    s = MixedSampler(full, terse, full_ratio=0.5, batch_size=3,
                     generator=torch.Generator().manual_seed(0), answer=qa, answer_ratio=0.5)
    kinds = [s.next() for _ in range(2000)]
    names = [k for k, _ in kinds]
    assert all(len(b) == 3 and all(e["density"] == "answer" for e in b)
               for k, b in kinds if k == "answer")
    assert abs(names.count("answer") / 2000 - 0.5) < 0.04
    assert abs(names.count("terse") / 2000 - 0.25) < 0.04
