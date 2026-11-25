import json

import torch

from conftest import make_tiny_tokenizer


def test_normalize_strips_case_punctuation_articles():
    from engram.eval.qa import normalize

    assert normalize("The  Eiffel Tower!") == "eiffel tower"


def test_em_and_f1_use_best_gold_answer():
    from engram.eval.qa import exact_match, f1_score

    golds = ["Denver Broncos", "the Broncos"]
    assert exact_match("broncos.", golds) == 1.0
    assert exact_match("Denver", golds) == 0.0
    assert abs(f1_score("Denver", golds) - 2 / 3) < 1e-9
    assert f1_score("Panthers", golds) == 0.0


def _examples(n=3):
    from engram.data.qa import make_qa_example

    tok = make_tiny_tokenizer()
    return [make_qa_example(tok, " ".join(f"w{(i + j) % 50}" for j in range(12)), "w1 w2",
                            f"w{i}", 8, 4, answers=[f"w{i}", f"w{i + 1}"]) for i in range(n)]


def test_make_qa_example_keeps_all_gold_answers():
    assert _examples(1)[0]["answers"] == ["w0", "w1"]


def test_generate_with_question_matches_loss_prompt(tiny_engram):
    eng = tiny_engram(chunk_size=8, k=4, control_vectors=True).eval()
    ex = _examples(1)[0]
    mem = eng.encode(ex["chunks"].reshape(1, -1))
    ctrl = eng._control("answer", 1)
    b = eng.decoder.generate(mem, ctrl, 4, eng.eos_id, question_ids=ex["question"][None])
    assert isinstance(b, list) and len(b) <= 4


def test_evaluate_qa_runs_all_conditions_and_writes_report(tiny_engram, tmp_path):
    from engram.eval.qa import evaluate_qa, qa_markdown, write_qa_report

    eng = tiny_engram(chunk_size=8, k=4, control_vectors=True)
    res = evaluate_qa(eng, _examples(3), max_new_tokens=4)
    for c in ("real", "shuffled", "zeroed", "base_zero_shot"):
        assert 0.0 <= res[c]["em"] <= 1.0 and 0.0 <= res[c]["f1"] <= 1.0
    assert res["n"] == 3
    assert "zero-shot" in qa_markdown([res])
    path = write_qa_report(tmp_path, res)
    assert json.loads(path.read_text())["n"] == 3
    assert (tmp_path / "eval" / "qa.md").exists()


def test_text_qa_layout_masks_paragraph_and_question_from_loss():
    from engram.eval.qa import text_qa_example

    tok = make_tiny_tokenizer()
    ex = _examples(1)[0]
    ids, labels = text_qa_example(tok, ex, eos_id=0)
    target = ex["target"].tolist()
    assert ids[-len(target):] == target
    assert labels[-len(target):] == target
    assert all(l == -100 for l in labels[: -len(target)])
    assert len(ids) == len(labels)


def test_train_text_qa_fits_answers_and_saves_lora_checkpoint(tmp_path):
    from conftest import make_tiny_model
    from engram.config import Config
    from engram.eval.qa import train_text_qa

    cfg = Config(chunk_size=8, lr=3e-3, text_qa_steps=15, batch_size=2, grad_accum=1,
                 warmup_steps=1)
    lm, losses = train_text_qa(make_tiny_model(), make_tiny_tokenizer(), cfg, _examples(3),
                               tmp_path)
    assert len(losses) == 15 and losses[-1] < losses[0]
    state = torch.load(tmp_path / "ckpt-15.pt")
    assert state and all("lora_" in k for k in state)


def test_evaluate_text_qa_scores_control_model(tmp_path):
    from conftest import make_tiny_model
    from engram.config import Config
    from engram.eval.qa import evaluate_text_qa, train_text_qa

    cfg = Config(chunk_size=8, text_qa_steps=1, batch_size=2, warmup_steps=1)
    lm, _ = train_text_qa(make_tiny_model(), make_tiny_tokenizer(), cfg, _examples(3), tmp_path)
    res = evaluate_text_qa(lm, make_tiny_tokenizer(), _examples(3), "cpu", max_new_tokens=4)
    assert res["n"] == 3 and 0.0 <= res["em"] <= 1.0 and 0.0 <= res["f1"] <= 1.0


def test_qa_report_shows_memory_over_control_ratio_when_control_exists(tmp_path):
    from engram.eval.qa import qa_markdown, write_qa_report
    from engram.eval.report import load_control_result

    res = {"k": 16, "n": 2, **{c: {"em": 0.2, "f1": 0.3} for c in ("real", "shuffled", "zeroed")}}
    assert "control" not in qa_markdown([res])
    assert load_control_result(tmp_path / "missing.json") is None
    cpath = tmp_path / "control.json"
    cpath.write_text(json.dumps({"n": 2, "em": 0.4, "f1": 0.6}))
    control = load_control_result(cpath)
    md = qa_markdown([res], control=control)
    assert "memory ÷ control F1" in md and "0.500" in md
    write_qa_report(tmp_path, res, control=control)
    saved = json.loads((tmp_path / "eval" / "qa.json").read_text())
    assert saved["text_qa_control"]["f1"] == 0.6 and abs(saved["memory_over_control_f1"] - 0.5) < 1e-9
