"""Stage 3b unattended pipeline (K=32):
warm-up -> recall gate -> gate summary -> joint stage -> text-QA control -> final QA/recall/density
evals -> final summary.

Usage: HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 uv run scripts/stage3b.py [--dir runs/stage3b] [--smoke]
                                                                   [--gate-threshold 0.25]

Everything goes under --dir: derived configs, warmup/<ts>, joint/<ts>, text-qa-<ts>, gate.{json,md},
final.{json,md} and state.json (finished stages). Re-running the same command resumes: finished
stages are skipped and an interrupted training stage resumes from its latest checkpoint. A gate
failure is final: re-running prints the same reason and stops.
--smoke: tiny step counts / eval sizes, default dir runs_smoke_k32/stage3b.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engram.eval.report import (RECALL_GATE, load_control_result, recall_gate, recall_gate_markdown,  # noqa: E402
                                stage3b_final, stage3b_final_markdown)

STAGES = ["warmup", "gate", "joint", "text_qa", "final_eval"]
GATE_N = 40  # SQuAD dev paragraphs for the recall gate (DECISIONS Q2)


def _load_state(pdir: Path) -> dict:
    f = pdir / "state.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _save_state(pdir: Path, state: dict) -> None:
    (pdir / "state.json").write_text(json.dumps(state, indent=2))


def _write_summary(pdir: Path, name: str, data: dict, md: str) -> None:
    (pdir / f"{name}.json").write_text(json.dumps(data, indent=2))
    (pdir / f"{name}.md").write_text(md + "\n")
    print(md, flush=True)


def run_pipeline(pdir, runners: dict, threshold: float = RECALL_GATE) -> str:
    """Runs the stages in order, recording each finished stage's output in <pdir>/state.json.
    Returns "done" or "gate_failed"."""
    pdir = Path(pdir)
    pdir.mkdir(parents=True, exist_ok=True)
    state = _load_state(pdir)

    def stage(name, *args):
        if name not in state:
            state[name] = runners[name](pdir, *args)
            _save_state(pdir, state)
        return state[name]

    warmup_run = stage("warmup")
    g = recall_gate(stage("gate", warmup_run), threshold)
    _write_summary(pdir, "gate", g, recall_gate_markdown(g))
    if not g["passed"]:
        print("STOP:", g["reason"], flush=True)
        return "gate_failed"
    joint_run = stage("joint", warmup_run)
    control = stage("text_qa")
    ev = stage("final_eval", joint_run)
    s = {"warmup_run": warmup_run, "joint_run": joint_run,
         **stage3b_final(ev["qa"], ev["recall"], ev["density"], ev["val"], control)}
    _write_summary(pdir, "final", s, stage3b_final_markdown(s))
    return "done"


# ---- real stage runners -------------------------------------------------------------------------

SMOKE = {
    "warmup": {"steps": 2, "val_interval": 2, "ckpt_interval": 2, "val_recall_samples": 2},
    "joint": {"steps": 2, "val_interval": 2, "ckpt_interval": 2, "val_gen_samples": 2,
              "val_recall_samples": 2},
    # eval sizes stay at cached sample counts (data cache is keyed by n; runs are HF-offline)
    "text_qa": {"text_qa_steps": 2},
    "eval_n": {"qa": 40, "recall": 2, "density": 32},
}


def _derive(src: str, dst: Path, overrides: dict) -> Path:
    d = yaml.safe_load(Path(src).read_text())
    d.update(overrides)
    dst.write_text(yaml.safe_dump(d))
    return dst


def _latest_ckpt(run: Path) -> Path | None:
    cks = sorted(run.glob("ckpt-*.pt"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    return cks[-1] if cks else None


def _train_stage(cfg_path: Path, out: Path, steps: int, init_from: str = "") -> str:
    """Trains into out/<ts>; resumes from the latest checkpoint of an earlier attempt."""
    import train

    runs = sorted(p for p in out.glob("*") if p.is_dir() and _latest_ckpt(p)) if out.exists() else []
    if runs:
        ck = _latest_ckpt(runs[-1])
        if ck.name != f"ckpt-{steps}.pt":
            train.main(["--config", str(cfg_path), "--resume", str(ck)])
        return str(runs[-1])
    for p in out.glob("*"):  # attempts that died before their first checkpoint
        if p.is_dir():
            p.rename(p.with_name(p.name + ".abandoned"))
    train.main(["--config", str(cfg_path), *(["--init-from", init_from] if init_from else [])])
    return str(next(p for p in sorted(out.glob("*")) if p.is_dir() and not p.name.endswith(".abandoned")))


def _eval(run: str, suite: str, n: int) -> dict:
    import eval as eval_script

    f = Path(run) / "eval" / f"{suite}.json"
    if not f.exists():
        eval_script.main(["--run", run, "--suite", suite, *(["--n", str(n)] if n else [])])
    return json.loads(f.read_text())


def make_runners(smoke: bool) -> dict:
    sm = SMOKE if smoke else {}
    eval_n = sm.get("eval_n", {})

    def warmup(pdir):
        over = {"run_dir": str(pdir / "warmup"), **sm.get("warmup", {})}
        cfg = _derive("configs/stage3b_warmup.yaml", pdir / "warmup.yaml", over)
        return _train_stage(cfg, pdir / "warmup", yaml.safe_load(cfg.read_text())["steps"])

    def gate(pdir, warmup_run):
        return _eval(warmup_run, "recall", eval_n.get("recall", GATE_N))

    def joint(pdir, warmup_run):
        over = {"run_dir": str(pdir / "joint"), "text_qa_result": str(pdir / "text_qa_control.json"),
                **sm.get("joint", {})}
        cfg = _derive("configs/stage3b_joint.yaml", pdir / "joint.yaml", over)
        steps = yaml.safe_load(cfg.read_text())["steps"]
        return _train_stage(cfg, pdir / "joint", steps, init_from=str(_latest_ckpt(Path(warmup_run))))

    def text_qa(pdir):
        import train_text_qa

        over = {"run_dir": str(pdir), "text_qa_result": str(pdir / "text_qa_control.json"),
                **sm.get("text_qa", {})}
        cfg = _derive("configs/text_qa_3b.yaml", pdir / "text_qa.yaml", over)
        train_text_qa.main(["--config", str(cfg)])  # cached result is reused on re-run
        return load_control_result(pdir / "text_qa_control.json")

    def final_eval(pdir, joint_run):
        rows = [json.loads(l) for l in (Path(joint_run) / "metrics.jsonl").read_text().splitlines() if l]
        val = next((r for r in reversed(rows) if "full_gap" in r), {})
        return {"qa": _eval(joint_run, "qa", eval_n.get("qa", 0)),
                "recall": _eval(joint_run, "recall", eval_n.get("recall", GATE_N)),
                "density": _eval(joint_run, "density", eval_n.get("density", 0)), "val": val}

    return {"warmup": warmup, "gate": gate, "joint": joint, "text_qa": text_qa, "final_eval": final_eval}


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="")
    ap.add_argument("--smoke", action="store_true", help="tiny step counts, output under runs_smoke_k32/")
    ap.add_argument("--gate-threshold", type=float, default=RECALL_GATE)
    args = ap.parse_args(argv)
    pdir = Path(args.dir or ("runs_smoke_k32/stage3b" if args.smoke else "runs/stage3b"))
    status = run_pipeline(pdir, make_runners(args.smoke), args.gate_threshold)
    print("pipeline:", status, "->", pdir)
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "done" else 1)
