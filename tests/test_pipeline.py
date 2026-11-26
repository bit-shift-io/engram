import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage3b.py"


def load_pipeline():
    spec = importlib.util.spec_from_file_location("stage3b", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stub_runners(calls, recall=0.5):
    def warmup(pdir):
        calls.append("warmup")
        return str(pdir / "warmup" / "run")

    def gate(pdir, warmup_run):
        calls.append("gate")
        return {"k": 32, "n": 40, "answer_recall": recall, "recon_token_f1": 0.4, "length_ratio": 0.9}

    def joint(pdir, warmup_run):
        calls.append("joint")
        return str(pdir / "joint" / "run")

    def text_qa(pdir):
        calls.append("text_qa")
        return {"n": 200, "em": 0.4, "f1": 0.5}

    def final_eval(pdir, joint_run):
        calls.append("final_eval")
        return {
            "qa": {"k": 32, "n": 200, "real": {"em": 0.2, "f1": 0.3}, "shuffled": {"em": 0.01, "f1": 0.05},
                   "zeroed": {"em": 0.0, "f1": 0.02}},
            "recall": {"k": 32, "n": 40, "answer_recall": 0.5, "recon_token_f1": 0.4, "length_ratio": 0.9},
            "density": {"terse_article_ratio": 0.1, "full_article_ratio": 0.9, "terse_rouge_l": 0.2},
            "val": {"full_gap": 1.2},
        }

    return {"warmup": warmup, "gate": gate, "joint": joint, "text_qa": text_qa, "final_eval": final_eval}


def test_gate_pass_runs_all_stages_and_writes_both_summaries(tmp_path):
    p = load_pipeline()
    calls = []
    status = p.run_pipeline(tmp_path, stub_runners(calls), threshold=0.25)
    assert status == "done"
    assert calls == ["warmup", "gate", "joint", "text_qa", "final_eval"]
    gate = json.loads((tmp_path / "gate.json").read_text())
    assert gate["passed"] and gate["answer_recall"] == 0.5
    final = json.loads((tmp_path / "final.json").read_text())
    assert final["qa"]["real"]["f1"] == 0.3
    assert final["memory_over_control_f1"] == pytest.approx(0.6)
    assert final["answer_recall"] == 0.5
    assert final["regressions"] == []
    md = (tmp_path / "final.md").read_text()
    assert "shuffled" in md and "squad_full_length_ratio" in md


def test_gate_failure_stops_before_joint_with_reason(tmp_path, capsys):
    p = load_pipeline()
    calls = []
    assert p.run_pipeline(tmp_path, stub_runners(calls, recall=0.1), threshold=0.25) == "gate_failed"
    assert calls == ["warmup", "gate"]
    gate = json.loads((tmp_path / "gate.json").read_text())
    assert not gate["passed"] and "0.100 < 0.25" in gate["reason"]
    assert "FAIL" in (tmp_path / "gate.md").read_text()
    assert "STOP" in capsys.readouterr().out
    assert not (tmp_path / "final.json").exists()


def test_rerun_skips_finished_stages(tmp_path):
    p = load_pipeline()
    calls = []
    runners = stub_runners(calls)

    def crash(pdir, warmup_run):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        p.run_pipeline(tmp_path, {**runners, "joint": crash})
    assert calls == ["warmup", "gate"]
    calls.clear()
    assert p.run_pipeline(tmp_path, runners) == "done"
    assert calls == ["joint", "text_qa", "final_eval"]
    calls.clear()
    assert p.run_pipeline(tmp_path, runners) == "done"
    assert calls == []
