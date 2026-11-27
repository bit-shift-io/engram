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
