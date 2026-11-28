"""Fact-QA unattended two-stage pipeline:
Stage A (density, init from the SQuAD cloze ckpt) -> cloze recall / density / val evals -> fact gate
-> Stage B (answer task) -> QA (n=200, as the text-QA control) / density / cloze recall evals -> summary.{json,md}.

Usage: HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 .venv/bin/python scripts/fact_qa.py [--dir runs/fact_qa] [--smoke]
                                                                         [--init-from <ckpt>]

Everything goes under --dir: derived configs, stage_a/<ts>, stage_b/<ts>, gate.{json,md}, summary.{json,md}
and state.json (finished stages). Re-running resumes: finished stages are skipped and an interrupted
training stage resumes from its latest checkpoint. A lost fact gate is final (status gate_failed, Stage B
never runs); a missed length band is only a warning and Stage B still runs.
--smoke: tiny step counts / eval sizes (sample counts unchanged: data caches are keyed by them),
default dir runs_smoke_fact_qa/. The summary records the QA n used.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage3b import _derive, _latest_ckpt, _train_stage  # noqa: E402

from engram.eval.report import (fact_gate, fact_gate_markdown, fact_qa_final,  # noqa: E402
                                fact_qa_final_markdown, load_control_result)

INIT_CKPT = "runs/squad_cloze/train/20261127-163240/ckpt-2000.pt"
CONTROL = "runs/text_qa_control.json"
QA_N = 200  # matches runs/text_qa_control.json (config qa_val_samples is 40)
FILES = {"cloze": "cloze_recall.json", "qa": "qa.json", "density": "density.json"}
SMOKE = {
    "stage_a": {"steps": 2, "val_interval": 2, "ckpt_interval": 2, "val_recall_samples": 2,
                "squad_cloze_val_max": 4},
    "stage_b": {"steps": 2, "val_interval": 2, "ckpt_interval": 2, "val_gen_samples": 2,
                "val_recall_samples": 2, "squad_cloze_val_max": 4},
    # QA n=40 is cached (the data cache is keyed by n); the real run uses QA_N=200 like the control
    "eval_n": {"cloze": 2, "qa": 40, "density": 4},
}


def _load_state(pdir: Path) -> dict:
    f = pdir / "state.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _save_state(pdir: Path, state: dict) -> None:
    (pdir / "state.json").write_text(json.dumps(state, indent=2))


def _write_summary(pdir: Path, name: str, data: dict, md: str) -> None:
    (pdir / f"{name}.json").write_text(json.dumps(data, indent=2))
    (pdir / f"{name}.md").write_text(md + "\n")
    print(md, flush=True)


def run_pipeline(pdir, runners: dict, ignore_gate: bool = False) -> str:
    """Returns "done" or "gate_failed". ignore_gate (smoke only) runs Stage B even when facts are lost,
    so the whole path is exercised; the gate result is still recorded."""
    pdir = Path(pdir)
    pdir.mkdir(parents=True, exist_ok=True)
    state = _load_state(pdir)

    def stage(name, *args):
        if name not in state:
            state[name] = runners[name](pdir, *args)
            _save_state(pdir, state)
        return state[name]

    a_run = stage("stage_a")
    a_ev = stage("eval_a", a_run)
    g = fact_gate(a_ev["cloze"], a_ev["density"], a_ev["val"])
    _write_summary(pdir, "gate", g, fact_gate_markdown(g))
    if not g["facts_passed"] and not ignore_gate:
        print("STOP: fact gate failed", flush=True)
        _write_summary(pdir, "summary", {"status": "gate_failed", "stage_a_run": a_run, "gate": g},
                       fact_gate_markdown(g) + "\n\nStatus: gate_failed (Stage B not run)")
        return "gate_failed"
    b_run = stage("stage_b", a_run)
    b_ev = stage("eval_b", b_run)
    s = {"status": "done", "stage_a_run": a_run, "stage_b_run": b_run, "gate": g,
         **fact_qa_final(b_ev["qa"], b_ev["cloze"], b_ev["density"], b_ev["val"], load_control_result(CONTROL))}
    md = fact_gate_markdown(g) + "\n\n" + fact_qa_final_markdown(s)
    _write_summary(pdir, "summary", s, md)
    return "done"


# ---- real stage runners -------------------------------------------------------------------------

def _eval(run: str, suite: str, n: int) -> dict:
    import eval as eval_script

    f = Path(run) / "eval" / FILES[suite]
    if not f.exists() or (suite == "qa" and json.loads(f.read_text()).get("n") != n):
        eval_script.main(["--run", run, "--suite", suite, *(["--n", str(n)] if n else [])])
    return json.loads(f.read_text())


def _last_val(run: str) -> dict:
    rows = [json.loads(l) for l in (Path(run) / "metrics.jsonl").read_text().splitlines() if l]
    return next((r for r in reversed(rows) if "full_gap" in r), {})


def make_runners(smoke: bool, init_from: str = INIT_CKPT) -> dict:
    sm = SMOKE if smoke else {}
    eval_n = sm.get("eval_n", {})

    def stage_a(pdir):
        over = {"run_dir": str(pdir / "stage_a"), **sm.get("stage_a", {})}
        cfg = _derive("configs/stage_a_density.yaml", pdir / "stage_a.yaml", over)
        return _train_stage(cfg, pdir / "stage_a", yaml.safe_load(cfg.read_text())["steps"], init_from=init_from)

    def eval_a(pdir, run):
        return {"cloze": _eval(run, "cloze", eval_n.get("cloze", 0)),
                "density": _eval(run, "density", eval_n.get("density", 0)), "val": _last_val(run)}

    def stage_b(pdir, a_run):
        over = {"run_dir": str(pdir / "stage_b"), **sm.get("stage_b", {})}
        cfg = _derive("configs/stage_b_answer.yaml", pdir / "stage_b.yaml", over)
        steps = yaml.safe_load(cfg.read_text())["steps"]
        return _train_stage(cfg, pdir / "stage_b", steps, init_from=str(_latest_ckpt(Path(a_run))))

    def eval_b(pdir, run):
        return {"qa": _eval(run, "qa", eval_n.get("qa", QA_N)),
                "cloze": _eval(run, "cloze", eval_n.get("cloze", 0))["real"],
                "density": _eval(run, "density", eval_n.get("density", 0)), "val": _last_val(run)}

    return {"stage_a": stage_a, "eval_a": eval_a, "stage_b": stage_b, "eval_b": eval_b}


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="")
    ap.add_argument("--smoke", action="store_true", help="tiny step counts, output under runs_smoke_fact_qa/")
    ap.add_argument("--init-from", default=INIT_CKPT)
    args = ap.parse_args(argv)
    pdir = Path(args.dir or ("runs_smoke_fact_qa" if args.smoke else "runs/fact_qa"))
    status = run_pipeline(pdir, make_runners(args.smoke, args.init_from), ignore_gate=args.smoke)
    print("pipeline:", status, "->", pdir)
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "done" else 1)
