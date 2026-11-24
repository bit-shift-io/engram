"""Usage: uv run scripts/generate.py "prompt" [--config configs/default.yaml] [--max-new-tokens N]"""

import argparse

import torch

from engram.config import load_config
from engram.model.base import generate, load_base_model


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)
    if cfg.device == "cpu":
        torch.set_num_threads(6)
    model, tok = load_base_model(cfg)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    new = generate(model, ids, device=cfg.device, max_new_tokens=args.max_new_tokens)
    print(args.prompt + tok.decode(new[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
