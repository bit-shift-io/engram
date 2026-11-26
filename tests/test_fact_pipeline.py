import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fact_objective.py"


def load_pipeline():
    spec = importlib.util.spec_from_file_location("fact_objective", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _recon(real, shuffled, zeroed):
    return {c: {"token_accuracy": 0.1, "rouge_l": v} for c, v in
            (("real", real), ("shuffled", shuffled), ("zeroed", zeroed))}


def stub_runners(calls, k16_recall=0.2, k16_real=0.3):
    def train(k):
        def run(pdir):
            calls.append(f"train_k{k}")
            return str(pdir / f"k{k}" / "run")
        return run

    def evaluate(k, recall, real):
        def run(pdir, run_dir):
            calls.append(f"eval_k{k}")
            return {"recall": {"k": k, "n": 40, "answer_recall": recall, "recon_token_f1": 0.3,
                               "length_ratio": 0.8},
                    "wikitext": _recon(real, 0.2, 0.05), "squad": _recon(0.15, 0.1, 0.02)}
        return run

    return {"train_k16": train(16), "eval_k16": evaluate(16, k16_recall, k16_real),
            "train_k32": train(32), "eval_k32": evaluate(32, 0.1, 0.25)}


def test_pass_stops_after_k16(tmp_path):
    p = load_pipeline()
    calls = []
    assert p.run_pipeline(tmp_path, stub_runners(calls)) == "passed_k16"
    assert calls == ["train_k16", "eval_k16"]
    s = json.loads((tmp_path / "summary.json").read_text())
    assert [a["k"] for a in s["arms"]] == [16]
    assert s["arms"][0]["passed"]
    md = (tmp_path / "summary.md").read_text()
    assert "0.050" in md and "0.226" in md  # both baselines shown


@pytest.mark.parametrize("recall,real", [(0.1, 0.3), (0.2, 0.15)])
def test_fail_runs_k32(tmp_path, recall, real):
    p = load_pipeline()
    calls = []
    assert p.run_pipeline(tmp_path, stub_runners(calls, recall, real)) == "done"
    assert calls == ["train_k16", "eval_k16", "train_k32", "eval_k32"]
    s = json.loads((tmp_path / "summary.json").read_text())
    assert [a["k"] for a in s["arms"]] == [16, 32]
    assert not s["arms"][0]["passed"] and s["arms"][0]["reason"]


def test_rerun_skips_finished_stages(tmp_path):
    p = load_pipeline()
    calls = []
    runners = stub_runners(calls, k16_recall=0.1)

    def crash(pdir):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        p.run_pipeline(tmp_path, {**runners, "train_k32": crash})
    assert calls == ["train_k16", "eval_k16"]
    calls.clear()
    assert p.run_pipeline(tmp_path, runners) == "done"
    assert calls == ["train_k32", "eval_k32"]
    calls.clear()
    p.run_pipeline(tmp_path, runners)
    assert calls == []
