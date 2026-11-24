"""Usage: uv run scripts/demo.py "<text>" [--run runs/<id>]

Prints input token count, memory vector count, compression ratio, and <full>/<terse> renders.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval import latest_checkpoint  # noqa: E402

from engram.config import load_config  # noqa: E402
from engram.model.base import load_base_model  # noqa: E402
from engram.model.engram import Engram  # noqa: E402
from engram.train.trainer import load_checkpoint  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--run", required=True, help="run dir with a Stage 2 checkpoint")
    ap.add_argument("--max-full-tokens", type=int, default=256,
                    help="cap on the <full> render (keeps a 512-token input under 30 s on CPU)")
    args = ap.parse_args(argv)
    run = Path(args.run)
    cfg = load_config(run / "config.yaml")
    torch.manual_seed(cfg.seed)
    if cfg.device == "cpu":
        torch.set_num_threads(cfg.num_threads)
    model, tok = load_base_model(cfg)
    eng = Engram(model, tok, cfg)
    load_checkpoint(eng, latest_checkpoint(run))
    eng.eval()
    n_tokens = len(tok(args.text, add_special_tokens=False).input_ids)
    n_mem = eng.encode(eng.tokenize_chunks(args.text)).shape[1]
    print(f"input tokens: {n_tokens}")
    print(f"memory vectors: {n_mem}")
    print(f"compression ratio: {n_tokens / n_mem:.1f}x")
    print(f"<full>: {eng.generate(args.text, 'full', args.max_full_tokens)}")
    print(f"<terse>: {eng.generate(args.text, 'terse', 64)}")


if __name__ == "__main__":
    main()
