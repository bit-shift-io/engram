"""Usage: uv run scripts/eval.py --run runs/<id> [runs/<id2> ...] [--suite reconstruction|density|qa|recall|cloze] [--n 32]

Evaluates each run's latest checkpoint, writes <run>/eval/reconstruction.{json,md},
and prints one comparison table (e.g. for the K sweep) across all runs.
"""

import argparse
import re
from pathlib import Path

import torch

from engram.config import load_config
from engram.data.reconstruction import load_chunks
from engram.data.qa import load_qa_examples
from engram.data.compression import load_terse_examples
from engram.eval.cloze_recall import (CLOZE_DEV_SAMPLES, cloze_recall_markdown, evaluate_cloze_recall,
                                       other_article_pool, select_recall_paragraphs,
                                       write_cloze_recall_report)
from engram.eval.density import density_markdown, evaluate_density, write_density_report
from engram.eval.reconstruct import evaluate_reconstruction
from engram.eval.qa import evaluate_qa, qa_markdown, write_qa_report
from engram.eval.recall import evaluate_recall, recall_markdown, write_recall_report
from engram.eval.report import load_control_result, markdown_table, write_run_report
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
    ap.add_argument("--suite", default="reconstruction", choices=["reconstruction", "density", "qa", "recall", "cloze"])
    ap.add_argument("--n", type=int, default=0, help="held-out chunks (default cfg.val_samples)")
    ap.add_argument("--eval-dir", default="eval", help="recall/cloze reports go to <run>/<eval-dir> (default eval)")
    ap.add_argument("--qa", action="store_true", help="same as --suite qa")
    args = ap.parse_args(argv)
    if args.qa:
        args.suite = "qa"

    results = []
    for run in map(Path, args.runs):
        cfg = load_config(run / "config.yaml")
        torch.manual_seed(cfg.seed)
        if cfg.device == "cpu":
            torch.set_num_threads(cfg.num_threads)
        model, tok = load_base_model(cfg)
        if args.suite == "qa":
            examples = load_qa_examples(cfg, tok, "validation", args.n or cfg.qa_eval_samples)
            eng = Engram(model, tok, cfg)
            load_checkpoint(eng, latest_checkpoint(run))
            result = evaluate_qa(eng, examples)
            write_qa_report(run, result, control=load_control_result(cfg.text_qa_result))
            results.append(result)
            continue
        if args.suite == "recall":
            # cache is keyed by n: load 200 dev examples and slice
            examples = load_qa_examples(cfg, tok, "validation", 200)[: args.n or 40]
            eng = Engram(model, tok, cfg)
            load_checkpoint(eng, latest_checkpoint(run))
            result = evaluate_recall(eng, examples)
            write_recall_report(run, result, args.eval_dir)
            results.append(result)
            continue
        if args.suite == "cloze":
            dev = load_qa_examples(cfg, tok, "validation", CLOZE_DEV_SAMPLES, with_title=True)  # cached
            picked = select_recall_paragraphs(tok, dev, 2 * (cfg.chunk_size + 1), args.n or 40)
            eng = Engram(model, tok, cfg)
            load_checkpoint(eng, latest_checkpoint(run))
            cloze = [c for _, c in picked]
            pool = other_article_pool(dev, {c["title"] for c in cloze}, len(cloze))
            result = evaluate_cloze_recall(eng, cloze, shuffle_pool=pool)
            write_cloze_recall_report(run, result, args.eval_dir)
            results.append(result)
            continue
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
    tables = {"density": density_markdown, "qa": qa_markdown, "recall": recall_markdown,
              "cloze": cloze_recall_markdown}
    print(tables.get(args.suite, markdown_table)(results))


if __name__ == "__main__":
    main()
