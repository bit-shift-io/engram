"""Does <full> reconstruction from memory still contain the SQuAD answer span?"""
import json
import sys

import torch

from engram.config import load_config
from engram.data.qa import load_qa_examples
from engram.eval.qa import f1_score, normalize
from engram.model.base import load_base_model
from engram.model.engram import Engram
from engram.train.trainer import load_checkpoint

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
ALL = {"stage2": "runs/20261125-163535/ckpt-600.pt", "stage3": "runs/20261126-072712/ckpt-640.pt"}
CKPTS = {k: ALL[k] for k in sys.argv[3].split(",")} if len(sys.argv) > 3 else ALL

cfg = load_config("runs/20261126-072712/config.yaml")
torch.manual_seed(0)
torch.set_num_threads(cfg.num_threads)
model, tok = load_base_model(cfg)
examples = load_qa_examples(cfg, tok, "validation", 200)[:N]
out = {}
for name, path in CKPTS.items():
    eng = Engram(model, tok, cfg)
    load_checkpoint(eng, path)
    eng.eval()
    hits, f1s, rows = 0, [], []
    for ex in examples:
        ctx = tok.decode(ex["chunks"].flatten(), skip_special_tokens=True)
        rec = eng.generate(ctx, "full")
        hit = any(normalize(a) and normalize(a) in normalize(rec) for a in ex["answers"])
        hits += hit
        f1s.append(f1_score(rec, [ctx]))
        rows.append({"answers": ex["answers"], "hit": hit, "rec": rec[:300]})
    out[name] = {"n": len(examples), "answer_in_recon": hits / len(examples),
                 "recon_token_f1": sum(f1s) / len(f1s), "rows": rows[:10]}
    print(name, out[name]["answer_in_recon"], out[name]["recon_token_f1"], flush=True)
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "answer_in_memory.json", "w"), indent=1)
