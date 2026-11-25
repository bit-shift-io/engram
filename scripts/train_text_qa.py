"""Usage: uv run scripts/train_text_qa.py --config configs/text_qa.yaml [--force] [--steps N] [--n N]

Text-QA control baseline: trains the LoRA, scores EM/F1 on the held-out SQuAD dev subset, writes
runs/text-qa-<ts>/{config.yaml, ckpt-<steps>.pt, eval/text_qa.json} and caches the result at
cfg.text_qa_result (reused by later runs; --force retrains)."""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import yaml

from engram.config import load_config
from engram.data.qa import load_qa_examples
from engram.eval.qa import evaluate_text_qa, train_text_qa
from engram.model.base import load_base_model


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/text_qa.yaml")
    ap.add_argument("--force", action="store_true", help="retrain even if a cached result exists")
    ap.add_argument("--steps", type=int, default=0, help="override text_qa_steps (smoke)")
    ap.add_argument("--n", type=int, default=0, help="override qa_eval_samples (smoke)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.steps:
        cfg = replace(cfg, text_qa_steps=args.steps)
    if args.n:
        cfg = replace(cfg, qa_eval_samples=args.n)
    cache = Path(cfg.text_qa_result)
    if cache.exists() and not args.force:
        print("cached control result:", cache, cache.read_text())
        return
    torch.manual_seed(cfg.seed)
    if cfg.device == "cpu":
        torch.set_num_threads(cfg.num_threads)
    model, tok = load_base_model(cfg)
    train = load_qa_examples(cfg, tok, "train", cfg.qa_train_samples)
    held_out = load_qa_examples(cfg, tok, "validation", cfg.qa_eval_samples)
    run_dir = Path(cfg.run_dir) / time.strftime("text-qa-%Y%m%d-%H%M%S")
    (run_dir / "eval").mkdir(parents=True)
    cfg_dict = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict))
    lm, losses = train_text_qa(model, tok, cfg, train, run_dir)
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps({"step": i + 1, "train_loss": l}) + "\n" for i, l in enumerate(losses)))
    result = {**evaluate_text_qa(lm, tok, held_out, cfg.device), "run": str(run_dir),
              "steps": cfg.text_qa_steps}
    (run_dir / "eval" / "text_qa.json").write_text(json.dumps(result, indent=2))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
