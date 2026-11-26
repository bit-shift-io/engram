"""Fact-objective unattended pipeline (Stage 1 + fact weighting + cloze task):
K=16 train -> recall + memory-use eval -> pass? stop : K=32 train -> same evals -> summary.

Usage: HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 .venv/bin/python scripts/fact_objective.py [--dir runs/fact_objective]
                                                                                    [--smoke] [--force-fail]

Everything goes under --dir: derived configs, k16/<ts>, k32/<ts>, summary.{json,md} and state.json
(finished stages). Eval results go to <run>/eval/fact_eval.json. Re-running the same command
resumes: finished stages are skipped and an interrupted training stage resumes from its latest
checkpoint.
Pass = answer recall >= 0.15 (40 SQuAD dev paragraphs) and WikiText val real ROUGE-L > shuffled.
--smoke: tiny step counts / eval sizes, default dir runs_smoke_fact/.
--force-fail: treat the K=16 arm as failed (exercise both arms).
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage3b import _derive, _latest_ckpt, _train_stage  # noqa: E402

from engram.eval.report import fact_arm, fact_summary_markdown  # noqa: E402

RECALL_N = 40  # SQuAD dev examples (sliced from the cached 200)
RECON_N = 15  # WikiText val chunks and SQuAD dev paragraph chunks (15 distinct in the cached 200)


def _load_state(pdir: Path) -> dict:
    f = pdir / "state.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _save_state(pdir: Path, state: dict) -> None:
    (pdir / "state.json").write_text(json.dumps(state, indent=2))


def run_pipeline(pdir, runners: dict, force_fail: bool = False) -> str:
    """Returns "passed_k16" (K=32 skipped) or "done" (both arms ran)."""
    pdir = Path(pdir)
    pdir.mkdir(parents=True, exist_ok=True)
    state = _load_state(pdir)

    def stage(name, *args):
        if name not in state:
            state[name] = runners[name](pdir, *args)
            _save_state(pdir, state)
        return state[name]

    arms = []
    for k in (16, 32):
        run = stage(f"train_k{k}")
        arm = fact_arm(k, run, stage(f"eval_k{k}", run))
        if k == 16 and force_fail and arm["passed"]:
            arm.update(passed=False, reason="forced fail (--force-fail)")
        arms.append(arm)
        if arm["passed"] and k == 16:
            break
    md = fact_summary_markdown(arms)
    (pdir / "summary.json").write_text(json.dumps({"arms": arms}, indent=2))
    (pdir / "summary.md").write_text(md + "\n")
    print(md, flush=True)
    return "passed_k16" if len(arms) == 1 else "done"


# ---- real stage runners -------------------------------------------------------------------------

SMOKE = {"train": {"steps": 2, "val_interval": 2, "ckpt_interval": 2, "val_recall_samples": 2},
         "recall_n": 2, "recon_n": 2}


def _evaluate(run: str, recall_n: int, recon_n: int) -> dict:
    import torch

    from engram.config import load_config
    from engram.data.qa import load_qa_examples
    from engram.data.reconstruction import load_chunks, squad_paragraph_chunks
    from engram.eval.recall import evaluate_recall
    from engram.eval.reconstruct import evaluate_reconstruction
    from engram.model.base import load_base_model
    from engram.model.engram import Engram
    from engram.train.trainer import load_checkpoint

    out = Path(run) / "eval" / "fact_eval.json"
    if out.exists():
        return json.loads(out.read_text())
    cfg = load_config(Path(run) / "config.yaml")
    torch.manual_seed(cfg.seed)
    if cfg.device == "cpu":
        torch.set_num_threads(cfg.num_threads)
    model, tok = load_base_model(cfg)
    eng = Engram(model, tok, cfg)
    load_checkpoint(eng, _latest_ckpt(Path(run)))
    dev = load_qa_examples(cfg, tok, "validation", 200)  # caches are keyed by n
    res = {"recall": evaluate_recall(eng, dev[:recall_n]),
           "wikitext": evaluate_reconstruction(eng, load_chunks(cfg, tok, "validation", cfg.val_samples)[:recon_n]),
           "squad": evaluate_reconstruction(eng, squad_paragraph_chunks(dev, recon_n))}
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    return res


def make_runners(smoke: bool) -> dict:
    sm = SMOKE if smoke else {}
    recall_n, recon_n = sm.get("recall_n", RECALL_N), sm.get("recon_n", RECON_N)

    def train(k):
        def run(pdir):
            over = {"run_dir": str(pdir / f"k{k}"), **sm.get("train", {})}
            cfg = _derive(f"configs/fact_k{k}.yaml", pdir / f"fact_k{k}.yaml", over)
            return _train_stage(cfg, pdir / f"k{k}", yaml.safe_load(cfg.read_text())["steps"])
        return run

    def evaluate(pdir, run):
        return _evaluate(run, recall_n, recon_n)

    return {"train_k16": train(16), "eval_k16": evaluate, "train_k32": train(32), "eval_k32": evaluate}


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="")
    ap.add_argument("--smoke", action="store_true", help="tiny step counts, output under runs_smoke_fact/")
    ap.add_argument("--force-fail", action="store_true", help="treat K=16 as failed so both arms run")
    args = ap.parse_args(argv)
    pdir = Path(args.dir or ("runs_smoke_fact" if args.smoke else "runs/fact_objective"))
    status = run_pipeline(pdir, make_runners(args.smoke), args.force_fail)
    print("pipeline:", status, "->", pdir)
    return status


if __name__ == "__main__":
    main()
