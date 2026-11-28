import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fact_qa  # noqa: E402


def _cloze(recall, gap):
    return {"real": {"cloze_recall": recall, "answer_loss": 1.0},
            "zeroed": {"cloze_recall": 0.0, "answer_loss": 1.0 + gap}}


def _runners(calls, recall=0.3, gap=2.0, density=None):
    dens = density if density is not None else {"terse_article_ratio": 0.1, "full_article_ratio": 0.9,
                                                "terse_rouge_l": 0.2}

    def rec(name, ret):
        def f(pdir, *a):
            calls.append(name)
            return ret
        return f

    qa = {"n": 200, **{c: {"em": 0.1, "f1": 0.2} for c in ("real", "shuffled", "zeroed")}}
    return {"stage_a": rec("stage_a", "A"),
            "eval_a": rec("eval_a", {"cloze": _cloze(recall, gap), "density": dens, "val": {"full_gap": 1.0}}),
            "stage_b": rec("stage_b", "B"),
            "eval_b": rec("eval_b", {"qa": qa, "cloze": {"cloze_recall": 0.2}, "density": dens,
                                     "val": {"full_gap": 1.0}})}


def test_gate_pass_runs_stage_b_and_writes_summary(tmp_path):
    calls = []
    assert fact_qa.run_pipeline(tmp_path, _runners(calls)) == "done"
    assert calls == ["stage_a", "eval_a", "stage_b", "eval_b"]
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["status"] == "done" and s["qa_n"] == 200
    assert (tmp_path / "summary.md").exists()


def test_gate_fail_stops_before_stage_b(tmp_path):
    calls = []
    assert fact_qa.run_pipeline(tmp_path, _runners(calls, recall=0.0)) == "gate_failed"
    assert "stage_b" not in calls
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "gate_failed"


def test_length_miss_still_runs_stage_b(tmp_path):
    calls = []
    bad = {"terse_article_ratio": 0.9, "full_article_ratio": 0.1, "terse_rouge_l": 0.2}
    assert fact_qa.run_pipeline(tmp_path, _runners(calls, density=bad)) == "done"
    assert "stage_b" in calls
    assert json.loads((tmp_path / "summary.json").read_text())["gate"]["length_warnings"]


def test_resume_skips_finished_stages(tmp_path):
    calls = []
    fact_qa.run_pipeline(tmp_path, _runners(calls))
    calls.clear()
    assert fact_qa.run_pipeline(tmp_path, _runners(calls)) == "done"
    assert calls == []


def test_ignore_gate_runs_stage_b_but_records_gate(tmp_path):
    calls = []
    assert fact_qa.run_pipeline(tmp_path, _runners(calls, recall=0.0), ignore_gate=True) == "done"
    assert "stage_b" in calls
    assert json.loads((tmp_path / "summary.json").read_text())["gate"]["facts_passed"] is False
