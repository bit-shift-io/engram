import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "squad_cloze.py"
CLOZE_BASE = {"k": 16, "n": 40, "real": {"hits": 5, "answer_loss": 3.48}, "shuffled": {"hits": 3, "answer_loss": 4.6},
        "zeroed": {"hits": 11, "answer_loss": 4.21}}
BASE = {"run": "runs/base", "answer_recall": 0.075, "cloze": CLOZE_BASE}


def load_pipeline():
    spec = importlib.util.spec_from_file_location("squad_cloze", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cloze(real, zeroed):
    return {"k": 16, "n": 40, "real": {"hits": 9, "hit_rate": 0.2, "answer_loss": real},
            "shuffled": {"hits": 3, "hit_rate": 0.1, "answer_loss": 4.5},
            "zeroed": {"hits": 5, "hit_rate": 0.1, "answer_loss": zeroed}}


def stub_runners(calls, recall=0.2, real=2.0, zeroed=4.0, early=None):
    def train(pdir):
        calls.append("train")
        return str(pdir / "train" / "run")

    def evaluate(pdir, run):
        calls.append("eval")
        return {"recall": {"k": 16, "n": 40, "answer_recall": recall, "recon_token_f1": 0.3, "length_ratio": 0.8},
                "cloze": cloze(real, zeroed), "early_stop": early}
    return {"train": train, "eval": evaluate}


def read(tmp_path):
    return json.loads((tmp_path / "summary.json").read_text())


def test_pass(tmp_path):
    p = load_pipeline()
    calls = []
    assert p.run_pipeline(tmp_path, stub_runners(calls), BASE) == "done"
    assert calls == ["train", "eval"]
    s = read(tmp_path)
    assert s["passed"] and all(c["passed"] for c in s["conditions"].values()) and not s["stopped_early"]
    md = (tmp_path / "summary.md").read_text()
    assert "3.48" in md and "4.21" in md and "5 / 3 / 11" in md and "0.075" in md


@pytest.mark.parametrize("kw,failed", [({"recall": 0.1}, "answer_recall"), ({"zeroed": 2.9}, "cloze_gap")])
def test_each_condition_fails_separately(tmp_path, kw, failed):
    p = load_pipeline()
    p.run_pipeline(tmp_path, stub_runners([], **kw), BASE)
    s = read(tmp_path)
    assert not s["passed"]
    assert {k for k, c in s["conditions"].items() if not c["passed"]} == {failed}
    assert "FAIL" in (tmp_path / "summary.md").read_text()


def test_early_stop_reason_is_recorded(tmp_path):
    p = load_pipeline()
    reason = "squad cloze real/zeroed val loss gap 0.050 < 0.1 at step 1000"
    p.run_pipeline(tmp_path, stub_runners([], real=4.0, zeroed=4.05, early={"step": 1000, "reason": reason}), BASE)
    s = read(tmp_path)
    assert s["stopped_early"] and s["stop_reason"] == reason and not s["passed"]
    assert reason in (tmp_path / "summary.md").read_text()


def test_rerun_resumes_from_state(tmp_path):
    p = load_pipeline()
    calls = []
    runners = stub_runners(calls)

    def crash(pdir, run):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        p.run_pipeline(tmp_path, {**runners, "eval": crash}, BASE)
    assert calls == ["train"]
    calls.clear()
    p.run_pipeline(tmp_path, runners, BASE)
    assert calls == ["eval"]
    calls.clear()
    p.run_pipeline(tmp_path, runners, BASE)
    assert calls == []


def test_markdown_renders_new_cloze_output_without_hits(tmp_path):
    from engram.eval.report import squad_cloze_markdown, squad_cloze_summary
    new = {"n": 40, **{k: {"blanks": 10, "cloze_recall": 0.3, "aligned_recall": 0.2, "answer_loss": 3.0}
                       for k in ("real", "shuffled", "zeroed")}}
    s = squad_cloze_summary("r", {"answer_recall": 0.2}, new, None, {"run": "b", "answer_recall": 0.1, "cloze": new})
    md = squad_cloze_markdown(s)
    assert "0.30" in md and "0.20" in md


# ---- rescore ------------------------------------------------------------------------------------

RESCORE = Path(__file__).resolve().parents[1] / "scripts" / "rescore_cloze.py"


def load_rescore():
    spec = importlib.util.spec_from_file_location("rescore_cloze", RESCORE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rs_result(recall, real, zeroed, cloze=0.0, aligned=0.0, zeroed_recall=0.0):
    cond = lambda loss, c=0.0, a=0.0: {"answer_loss": loss, "cloze_recall": c, "aligned_recall": a}  # noqa: E731
    return {"recall": {"answer_recall": recall, "zeroed_answer_recall": zeroed_recall},
            "cloze": {"real": cond(real, cloze, aligned), "shuffled": cond(5.0), "zeroed": cond(zeroed)}}


OLD = {"base": {"answer_recall": 0.075, "gap": 1.0}, "new": {"answer_recall": 0.2, "gap": 1.7}}


def run_rescore(tmp_path, base, new):
    p = load_rescore()
    results = {"base": base, "new": new}
    p.run_rescore(tmp_path, {"eval": lambda pdir, name, run: results[name]}, {"base": "rb", "new": "rn"}, OLD)
    return json.loads((tmp_path / "summary.json").read_text())


def test_rescore_pass(tmp_path):
    s = run_rescore(tmp_path, rs_result(0.1, 3.0, 4.0), rs_result(0.25, 2.0, 4.0))
    assert s["rule"]["answer_recall"]["passed"] and s["rule"]["gap"]["passed"] and s["passed"]
    assert s["checkpoints"]["base"]["old"]["answer_recall"] == 0.075
    assert s["checkpoints"]["new"]["rescored"]["gap"] == 2.0
    assert "0.075" in (tmp_path / "summary.md").read_text()


def test_rescore_fail_on_recall(tmp_path):
    s = run_rescore(tmp_path, rs_result(0.1, 3.0, 4.0), rs_result(0.15, 2.0, 4.0))
    assert not s["rule"]["answer_recall"]["passed"] and s["rule"]["gap"]["passed"] and not s["passed"]


def test_rescore_fail_on_gap(tmp_path):
    s = run_rescore(tmp_path, rs_result(0.1, 3.0, 4.0), rs_result(0.25, 3.5, 4.0))
    assert s["rule"]["answer_recall"]["passed"] and not s["rule"]["gap"]["passed"]


def test_rescore_zero_baseline_is_undefined_not_pass(tmp_path):
    s = run_rescore(tmp_path, rs_result(0.0, 3.0, 4.0), rs_result(0.25, 2.0, 4.0))
    assert s["rule"]["answer_recall"]["passed"] is None and not s["passed"]
    assert "undefined" in (tmp_path / "summary.md").read_text()


def test_rescore_markdown_shows_cloze_and_aligned_recall_per_checkpoint(tmp_path):
    run_rescore(tmp_path, rs_result(0.1, 3.0, 4.0, 0.041, 0.023), rs_result(0.25, 2.0, 4.0, 0.188, 0.056))
    md = (tmp_path / "summary.md").read_text()
    assert "aligned recall" in md and "0.188" in md and "0.056" in md and "0.041" in md


def test_rescore_reports_zeroed_answer_recall(tmp_path):
    s = run_rescore(tmp_path, rs_result(0.175, 3.0, 4.0, zeroed_recall=0.125), rs_result(0.175, 2.0, 4.0, zeroed_recall=0.05))
    assert s["checkpoints"]["base"]["rescored"]["zeroed_answer_recall"] == 0.125
    md = (tmp_path / "summary.md").read_text()
    assert "zeroed answer recall" in md and "0.125" in md and "0.050" in md
