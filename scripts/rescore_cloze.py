"""Re-score the baseline and SQuAD cloze checkpoints with the fixed answer recall + cloze recall evals.

Usage: HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 .venv/bin/python scripts/rescore_cloze.py [--dir runs/cloze_rescore]

New reports go to <run>/eval_rescore/ (original eval/ reports are untouched). Writes
<dir>/summary.{json,md}. Pass rule (each flagged separately): the SQuAD cloze checkpoint's answer
recall >= 2x baseline AND its real-minus-zeroed cloze loss gap >= 2x baseline gap. A baseline of 0 makes
the rule undefined, not a pass. Takes ~25 min. Re-running reuses finished reports in eval_rescore/.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engram.eval.report import rescore_markdown, rescore_summary  # noqa: E402

RUNS = {"base": "runs/fact_objective/k16/20261126-212058", "new": "runs/squad_cloze/train/20261127-163240"}
EVAL_DIR = "eval_rescore"


def old_scores(runs: dict) -> dict:
    """Numbers from the original reports (baseline: fact_eval.json; SQuAD cloze: recall.json)."""
    out = {}
    for name, run in runs.items():
        ev = Path(run) / "eval"
        rf = ev / "recall.json" if (ev / "recall.json").exists() else ev / "fact_eval.json"
        rec = json.loads(rf.read_text())
        rec = rec.get("recall", rec)
        cl = json.loads((ev / "cloze_recall.json").read_text())
        out[name] = {"answer_recall": rec["answer_recall"],
                     "gap": cl["zeroed"]["answer_loss"] - cl["real"]["answer_loss"]}
    return out


def run_rescore(pdir, runners: dict, runs: dict, old: dict) -> str:
    pdir = Path(pdir)
    pdir.mkdir(parents=True, exist_ok=True)
    rescored = {name: runners["eval"](pdir, name, run) for name, run in runs.items()}
    s = rescore_summary(old, rescored)
    md = rescore_markdown(s)
    (pdir / "summary.json").write_text(json.dumps(s, indent=2))
    (pdir / "summary.md").write_text(md + "\n")
    print(md, flush=True)
    return "done"


def make_runners() -> dict:
    def evaluate(pdir, name, run):
        import eval as eval_script

        r = Path(run)
        for suite, f in (("recall", "recall"), ("cloze", "cloze_recall")):
            if not (r / EVAL_DIR / f"{f}.json").exists():
                eval_script.main(["--run", run, "--suite", suite, "--eval-dir", EVAL_DIR])
        return {"recall": json.loads((r / EVAL_DIR / "recall.json").read_text()),
                "cloze": json.loads((r / EVAL_DIR / "cloze_recall.json").read_text())}
    return {"eval": evaluate}


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/cloze_rescore")
    args = ap.parse_args(argv)
    return run_rescore(args.dir, make_runners(), RUNS, old_scores(RUNS))


if __name__ == "__main__":
    main()
