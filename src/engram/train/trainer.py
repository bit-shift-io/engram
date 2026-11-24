"""Training loop: grad accumulation, cosine LR, validation, checkpoints, resume.

Stage 1 trains <full> reconstruction only. Stage 2 (full_ratio < 1) mixes in <terse> summary
batches, can start from a Stage 1 checkpoint (init_from) and train the encoder at a lower LR.
"""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from engram.config import Config
from engram.data.mixed import MixedSampler
from engram.train.losses import reconstruction_loss, validation_loss


def lr_at(step: int, cfg: Config) -> float:
    """Linear warmup then cosine decay to 0 over cfg.steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _is_encoder_param(name: str) -> bool:
    return name.startswith("encoder.") or ("lora_" in name and ".encoder." in name)


def param_groups(eng, cfg: Config) -> list[dict]:
    """[encoder LoRA + memory queries + projection (scaled LR), decoder LoRA + control embeddings].
    With encoder_lr_scale == 1 a single group is used, keeping Stage 1 checkpoints resumable."""
    if cfg.encoder_lr_scale == 1.0:
        return [{"params": eng.trainable_parameters(), "scale": 1.0}]
    enc, dec = [], []
    for n, p in eng.named_parameters():
        if p.requires_grad:
            (enc if _is_encoder_param(n) else dec).append(p)
    return [{"params": enc, "scale": cfg.encoder_lr_scale}, {"params": dec, "scale": 1.0}]


def _trainable_state(eng) -> dict:
    return {n: p.detach().clone() for n, p in eng.named_parameters() if p.requires_grad}


def save_checkpoint(eng, opt, step: int, path: Path) -> None:
    torch.save({"step": step, "params": _trainable_state(eng), "opt": opt.state_dict()}, path)


def load_checkpoint(eng, path, opt=None) -> int:
    """Load trainable params (and optimizer if given); returns the saved step."""
    ck = torch.load(path)
    named = dict(eng.named_parameters())
    with torch.no_grad():
        for n, v in ck["params"].items():
            named[n].copy_(v)
    if opt is not None:
        opt.load_state_dict(ck["opt"])
    return ck["step"]


def _batch_loss(eng, density: str, batch) -> torch.Tensor:
    return reconstruction_loss(eng, batch) if density == "full" else eng.terse_loss(batch)


def train(eng, cfg: Config, train_chunks: torch.Tensor, val_chunks: torch.Tensor,
          terse_train: list[dict] | None = None, terse_val: list[dict] | None = None) -> Path:
    if cfg.resume:
        run_dir = Path(cfg.resume).parent
    else:
        run_dir = Path(cfg.run_dir) / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))
    params = eng.trainable_parameters()
    opt = torch.optim.Adam(param_groups(eng, cfg), lr=cfg.lr)
    if cfg.resume:
        start = load_checkpoint(eng, cfg.resume, opt)
    else:
        if cfg.init_from:
            load_checkpoint(eng, cfg.init_from)  # params only: fresh optimizer and step
        start = 0
    gen = torch.Generator().manual_seed(cfg.seed + start)
    sampler = MixedSampler(train_chunks, terse_train, cfg.full_ratio, cfg.batch_size, gen)
    eng.train()
    metrics = open(run_dir / "metrics.jsonl", "a")

    def log(**kw):
        metrics.write(json.dumps(kw) + "\n")
        metrics.flush()

    for step in range(start, cfg.steps):
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg) * g["scale"]
        opt.zero_grad()
        total, counts = 0.0, {"full": 0, "terse": 0}
        for _ in range(cfg.grad_accum):
            density, batch = sampler.next()
            counts[density] += 1
            loss = _batch_loss(eng, density, batch) / cfg.grad_accum
            loss.backward()
            total += float(loss)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
        opt.step()
        n = step + 1
        extra = {}
        if len(opt.param_groups) > 1:
            extra = {"encoder_lr": opt.param_groups[0]["lr"]}
        log(step=n, train_loss=total, lr=opt.param_groups[-1]["lr"], grad_norm=gnorm,
            n_full=counts["full"], n_terse=counts["terse"], secs=time.time() - t0, **extra)
        if n % cfg.val_interval == 0 or n == cfg.steps:
            row = {"val_loss": validation_loss(eng, val_chunks, cfg.batch_size)}
            if terse_val:
                row["terse_val_loss"] = _terse_val_loss(eng, terse_val, cfg.batch_size)
            log(step=n, **row)
        if n % cfg.ckpt_interval == 0 or n == cfg.steps:
            save_checkpoint(eng, opt, n, run_dir / f"ckpt-{n}.pt")
    metrics.close()
    return run_dir


@torch.no_grad()
def _terse_val_loss(eng, examples: list[dict], batch_size: int) -> float:
    eng.eval()
    total = 0.0
    for i in range(0, len(examples), batch_size):
        b = examples[i : i + batch_size]
        total += float(eng.terse_loss(b)) * len(b)
    eng.train()
    return total / len(examples)
