import json

from conftest import make_tiny_tokenizer


def test_answer_in_text_normalised_multi_gold_and_empty():
    from engram.eval.recall import answer_in_text

    assert answer_in_text(["Denver Broncos"], "The denver  broncos, won.")
    assert answer_in_text(["nope", "the Broncos"], "go broncos")
    assert not answer_in_text(["Panthers"], "go broncos")
    assert not answer_in_text([], "go broncos")
    assert not answer_in_text(["", "the"], "go broncos")


def test_evaluate_recall_metrics_in_range_and_report(tiny_engram, tmp_path):
    from engram.data.qa import make_qa_example
    from engram.eval.recall import evaluate_recall, recall_markdown, write_recall_report

    tok = make_tiny_tokenizer()
    exs = [make_qa_example(tok, " ".join(f"w{(i + j) % 50}" for j in range(12)), "w1 w2",
                           f"w{i}", 8, 4, answers=[f"w{i}"]) for i in range(2)]
    eng = tiny_engram(chunk_size=8, k=4, control_vectors=True)
    res = evaluate_recall(eng, exs)
    assert res["n"] == 2 and res["k"] == 4
    assert 0.0 <= res["answer_recall"] <= 1.0
    assert 0.0 <= res["recon_token_f1"] <= 1.0
    assert res["length_ratio"] >= 0.0
    assert "answer recall" in recall_markdown([res])
    path = write_recall_report(tmp_path, res)
    assert json.loads(path.read_text())["n"] == 2
    assert (tmp_path / "eval" / "recall.md").exists()


def test_answer_in_text_requires_word_boundaries():
    from engram.eval.recall import answer_in_text

    assert not answer_in_text(["2"], "2015")
    assert answer_in_text(["2"], "2 sacks")
    assert answer_in_text(["Cam Newton"], "cam newton won")
    assert answer_in_text(["$5 million"], "paid $5 million.")


def test_evaluate_recall_reports_zeroed_memory_control(tiny_engram):
    from engram.data.qa import make_qa_example
    from engram.eval.recall import evaluate_recall

    tok = make_tiny_tokenizer()
    exs = [make_qa_example(tok, " ".join(f"w{(i + j) % 50}" for j in range(12)), "w1 w2",
                           f"w{i}", 8, 4, answers=[f"w{i}"]) for i in range(2)]
    eng = tiny_engram(chunk_size=8, k=4, control_vectors=True)
    seen = []
    orig = eng.decoder.generate
    eng.decoder.generate = lambda memory, *a, **k: (seen.append(float(memory.abs().sum())), orig(memory, *a, **k))[1]
    res = evaluate_recall(eng, exs)
    assert 0.0 <= res["zeroed_answer_recall"] <= 1.0
    assert seen.count(0.0) == 2 and len(seen) == 4  # one real and one zeroed pass per paragraph
