"""Usage: uv run scripts/eval.py --run runs/<id> [runs/<id2> ...] [--suite reconstruction] [--n 32]

Evaluates each run's latest checkpoint, writes <run>/eval/reconstruction.{json,md},
and prints one comparison table (e.g. for the K sweep) across all runs.
"""

import argparse
import re
from pathlib import Path

import torch

from engram.config import load_config
from engram.data.reconstruction import load_chunks
from engram.data.compression import load_terse_examples
from engram.eval.density import density_markdown, evaluate_density, write_density_report
from engram.eval.reconstruct import evaluate_reconstruction
from engram.eval.report import markdown_table, write_run_report
from engram.model.base import load_base_model
from engram.model.engram import Engram
from engram.train.trainer import load_checkpoint


def latest_checkpoint(run_dir: Path) -> Path:
    cks = sorted(run_dir.glob("ckpt-*.pt"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if not cks:
        raise FileNotFoundError(f"no ckpt-N.pt in {run_dir}")
    return cks[-1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", "--runs", dest="runs", nargs="+", required=True)
    ap.add_argument("--suite", default="reconstruction", choices=["reconstruction", "density"])
    ap.add_argument("--n", type=int, default=0, help="held-out chunks (default cfg.val_samples)")
    args = ap.parse_args(argv)

    results = []
    for run in map(Path, args.runs):
        cfg = load_config(run / "config.yaml")
        torch.manual_seed(cfg.seed)
        if cfg.device == "cpu":
            torch.set_num_threads(cfg.num_threads)
        model, tok = load_base_model(cfg)
        if args.suite == "density":
            examples = load_terse_examples(cfg, tok, "validation", args.n or cfg.terse_val_samples)
            eng = Engram(model, tok, cfg)
            load_checkpoint(eng, latest_checkpoint(run))
            result = evaluate_density(eng, examples)
            write_density_report(run, result)
            results.append(result)
            continue
        chunks = load_chunks(cfg, tok, "validation", args.n or cfg.val_samples)
        eng = Engram(model, tok, cfg)
        load_checkpoint(eng, latest_checkpoint(run))
        result = evaluate_reconstruction(eng, chunks)
        write_run_report(run, result)
        results.append(result)
    print(density_markdown(results) if args.suite == "density" else markdown_table(results))


if __name__ == "__main__":
    main()
