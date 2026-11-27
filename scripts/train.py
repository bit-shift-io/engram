"""Usage: uv run scripts/train.py --config configs/stage1.yaml [--resume runs/<ts>/ckpt-N.pt]
                                  [--init-from runs/<stage1>/ckpt-N.pt]"""

import argparse
from dataclasses import replace

import torch

from engram.config import load_config
from engram.data.compression import load_terse_examples
from engram.data.qa import load_qa_examples
from engram.eval.cloze_recall import CLOZE_DEV_SAMPLES
from engram.data.reconstruction import load_chunks, mix_chunks
from engram.model.base import load_base_model
from engram.model.engram import Engram
from engram.train.trainer import train


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--resume", default="")
    ap.add_argument("--init-from", default="")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.resume:
        cfg = replace(cfg, resume=args.resume)
    if args.init_from:
        cfg = replace(cfg, init_from=args.init_from)
    torch.manual_seed(cfg.seed)
    if cfg.device == "cpu":
        torch.set_num_threads(cfg.num_threads)
    model, tok = load_base_model(cfg)
    train_chunks = load_chunks(cfg, tok, "train", cfg.train_samples)
    val_chunks = load_chunks(cfg, tok, "validation", cfg.val_samples)
    terse_train = terse_val = None
    if cfg.full_ratio < 1.0:
        terse_train = load_terse_examples(cfg, tok, "train", cfg.terse_train_samples)
        terse_val = load_terse_examples(cfg, tok, "validation", cfg.terse_val_samples)
    answer_train = answer_val = cloze_dev = None
    if cfg.squad_recon_share > 0:
        train_chunks = mix_chunks(train_chunks, load_qa_examples(cfg, tok, "train", cfg.qa_train_samples),
                                  cfg.squad_recon_share, cfg.seed)
    if cfg.answer_ratio > 0 or cfg.squad_cloze_ratio > 0:
        answer_train = load_qa_examples(cfg, tok, "train", cfg.qa_train_samples)
    if cfg.answer_ratio > 0 or cfg.val_recall_samples > 0:
        answer_val = load_qa_examples(cfg, tok, "validation", cfg.qa_val_samples)
    if cfg.squad_cloze_ratio > 0:  # the cloze recall eval picks its 40 paragraphs from these
        cloze_dev = load_qa_examples(cfg, tok, "validation", CLOZE_DEV_SAMPLES)
    eng = Engram(model, tok, cfg)
    print("run dir:", train(eng, cfg, train_chunks, val_chunks, terse_train, terse_val,
                            answer_train, answer_val, cloze_dev))


if __name__ == "__main__":
    main()
