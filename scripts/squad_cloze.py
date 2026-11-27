"""SQuAD cloze unattended pipeline: train (fact K=16 ckpt-3000 + SQuAD cloze) -> answer recall eval ->
cloze recall eval -> summary.{json,md} with the pass rule and the fact K=16 baseline row.

Usage: HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 .venv/bin/python scripts/squad_cloze.py [--dir runs/squad_cloze] [--smoke]

Everything goes under --dir: derived config, train/<ts>, summary.{json,md} and state.json (finished
stages). Re-running the same command resumes: finished stages are skipped and an interrupted
training stage resumes from its latest checkpoint. The trainer stops early at step 1000 if the SQuAD
cloze val losses (real vs zeroed) differ by < 0.1 nat; the summary records that.
Pass = answer recall >= 0.15 AND cloze real beats zeroed answer loss by >= 1 nat (each reported).
--smoke: tiny step counts / eval sizes, default dir runs_smoke_squad_cloze/.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage3b import _derive, _latest_ckpt, _train_stage  # noqa: E402

from engram.eval.report import baseline_row, squad_cloze_markdown, squad_cloze_summary  # noqa: E402

CONFIG = "configs/squad_cloze_k16.yaml"
RECALL_N = 40
SMOKE = {"train": {"steps": 4, "val_interval": 2, "ckpt_interval": 2, "val_recall_samples": 2,
                   "early_stop_step": 0},
         "n": 2}


def _load_state(pdir: Path) -> dict:
    f = pdir / "state.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _save_state(pdir: Path, state: dict) -> None:
    (pdir / "state.json").write_text(json.dumps(state, indent=2))


def run_pipeline(pdir, runners: dict, baseline: dict | None) -> str:
    pdir = Path(pdir)
    pdir.mkdir(parents=True, exist_ok=True)
    state = _load_state(pdir)

    def stage(name, *args):
        if name not in state:
            state[name] = runners[name](pdir, *args)
            _save_state(pdir, state)
        return state[name]

    run = stage("train")
    ev = stage("eval", run)
    s = squad_cloze_summary(run, ev["recall"], ev["cloze"], ev.get("early_stop"), baseline)
    md = squad_cloze_markdown(s)
    (pdir / "summary.json").write_text(json.dumps(s, indent=2))
    (pdir / "summary.md").write_text(md + "\n")
    print(md, flush=True)
    return "done"


# ---- real stage runners -------------------------------------------------------------------------

def make_runners(smoke: bool) -> dict:
    sm = SMOKE if smoke else {}
    n = sm.get("n", 0)

    def train(pdir):
        over = {"run_dir": str(pdir / "train"), **sm.get("train", {})}
        cfg = _derive(CONFIG, pdir / "squad_cloze.yaml", over)
        steps = yaml.safe_load(cfg.read_text())["steps"]
        out = pdir / "train"
        stopped = [p for p in out.glob("*/eval/early_stop.json")] if out.exists() else []
        if stopped:  # early-stopped on an earlier attempt: do not resume to the full step count
            return str(stopped[-1].parent.parent)
        return _train_stage(cfg, out, steps)

    def evaluate(pdir, run):
        import eval as eval_script

        r = Path(run)
        for suite, name in (("recall", "recall"), ("cloze", "cloze_recall")):
            if not (r / "eval" / f"{name}.json").exists():
                eval_script.main(["--run", run, "--suite", suite, *(["--n", str(n)] if n else [])])
        es = r / "eval" / "early_stop.json"
        return {"recall": json.loads((r / "eval" / "recall.json").read_text()),
                "cloze": json.loads((r / "eval" / "cloze_recall.json").read_text()),
                "early_stop": json.loads(es.read_text()) if es.exists() else None}

    return {"train": train, "eval": evaluate}


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="")
    ap.add_argument("--smoke", action="store_true", help="tiny step counts, output under runs_smoke_squad_cloze/")
    args = ap.parse_args(argv)
    pdir = Path(args.dir or ("runs_smoke_squad_cloze" if args.smoke else "runs/squad_cloze"))
    status = run_pipeline(pdir, make_runners(args.smoke), baseline_row())
    print("pipeline:", status, "->", pdir)
    return status


if __name__ == "__main__":
    main()
