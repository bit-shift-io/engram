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
from engram.data.compression import full_example
from engram.data.mixed import MixedSampler
from engram.eval.density import evaluate_density
from engram.eval.qa import evaluate_qa
from engram.eval.recall import evaluate_recall
from engram.eval.report import load_control_result, stage3_gates, stage3_markdown
from engram.train.losses import reconstruction_loss, validation_loss


def lr_at(step: int, cfg: Config) -> float:
    """Linear warmup then cosine decay to 0 over cfg.steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _is_encoder_param(name: str) -> bool:
    return (name.startswith("encoder.") or name.startswith("decoder.mem_adapter.")
            or ("lora_" in name and ".encoder." in name))


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
    buffers = {n: b.detach().clone() for n, b in eng.encoder.proj.named_buffers()}
    torch.save({"step": step, "params": _trainable_state(eng), "proj_buffers": buffers,
                "opt": opt.state_dict()}, path)


_CONTROL_ROWS = ("control.weight", "decoder.ctrl_vec.vec")  # one row per CONTROL_TOKENS entry


def load_checkpoint(eng, path, opt=None) -> int:
    """Load trainable params (and optimizer if given); returns the saved step."""
    ck = torch.load(path)
    named = dict(eng.named_parameters())
    with torch.no_grad():
        for n, v in ck["params"].items():
            p = named[n]
            if n in _CONTROL_ROWS and v.shape[0] < p.shape[0] and v.shape[1:] == p.shape[1:]:
                # Checkpoint from before a control token was appended (e.g. Stage 2 without
                # <answer>): keep existing flags; new flag vectors start at zero (a no-op).
                p[: v.shape[0]].copy_(v)
                if n == "decoder.ctrl_vec.vec":
                    p[v.shape[0]:].zero_()
                continue
            p.copy_(v)
        bufs = dict(eng.encoder.proj.named_buffers())
        for n, v in ck.get("proj_buffers", {}).items():
            if n in bufs:
                bufs[n].copy_(v)
    if opt is not None:
        opt.load_state_dict(ck["opt"])
    return ck["step"]


def _batch_loss(eng, density: str, batch) -> torch.Tensor:
    # "multi" (multi-chunk <full>), "terse" and "answer" batches are lists of examples with their own density.
    return reconstruction_loss(eng, batch) if density == "full" else eng.terse_loss(batch)


def train(eng, cfg: Config, train_chunks: torch.Tensor, val_chunks: torch.Tensor,
          terse_train: list[dict] | None = None, terse_val: list[dict] | None = None,
          answer_train: list[dict] | None = None, answer_val: list[dict] | None = None) -> Path:
    if cfg.resume:
        run_dir = Path(cfg.resume).parent
    else:
        run_dir = Path(cfg.run_dir) / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True)
        cfg_dict = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()}
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict))
    params = eng.trainable_parameters()
    opt = torch.optim.Adam(param_groups(eng, cfg), lr=cfg.lr)
    if cfg.resume:
        start = load_checkpoint(eng, cfg.resume, opt)
    else:
        if cfg.init_from:
            load_checkpoint(eng, cfg.init_from)  # params only: fresh optimizer and step
        start = 0
    gen = torch.Generator().manual_seed(cfg.seed + start)
    multi = None
    if cfg.multi_full_ratio > 0:
        multi = [full_example(e, eng.eos_id) for e in terse_train]
    squad_full = None
    if cfg.squad_full_ratio > 0:
        squad_full = _squad_paragraph_examples(answer_train, eng.eos_id)
    sampler = MixedSampler(train_chunks, terse_train, cfg.full_ratio, cfg.batch_size, gen,
                           multi_full=multi, multi_ratio=cfg.multi_full_ratio,
                           answer=answer_train, answer_ratio=cfg.answer_ratio,
                           squad_full=squad_full, squad_ratio=cfg.squad_full_ratio)
    eng.train()
    metrics = open(run_dir / "metrics.jsonl", "a")

    def log(**kw):
        metrics.write(json.dumps(kw) + "\n")
        metrics.flush()

    last_val = None
    for step in range(start, cfg.steps):
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg) * g["scale"]
        opt.zero_grad()
        total, counts = 0.0, {"full": 0, "multi": 0, "squad_full": 0, "terse": 0, "answer": 0}
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
            n_full=counts["full"], n_multi=counts["multi"], n_squad_full=counts["squad_full"], n_terse=counts["terse"],
            n_answer=counts["answer"], secs=time.time() - t0, **extra)
        if n % cfg.val_interval == 0 or n == cfg.steps:
            row = {"val_loss": validation_loss(eng, val_chunks, cfg.batch_size)}
            if terse_val:
                row["terse_val_loss"] = _terse_val_loss(eng, terse_val, cfg.batch_size)
                if multi:
                    full_val = [full_example(e, eng.eos_id) for e in terse_val]
                    row["multi_val_loss"] = _terse_val_loss(eng, full_val, cfg.batch_size)
            if answer_val:
                row["answer_val_loss"] = _terse_val_loss(eng, answer_val, cfg.batch_size)
            if cfg.val_gen_samples and answer_val:
                row.update(_generation_val(eng, cfg, val_chunks, terse_val, answer_val))
            if cfg.val_recall_samples and answer_val:
                rec = evaluate_recall(eng, answer_val[: cfg.val_recall_samples])
                row.update({"answer_recall": rec["answer_recall"],
                            "recon_token_f1": rec["recon_token_f1"],
                            "squad_full_length_ratio": rec["length_ratio"]})
                eng.train()
            log(step=n, **row)
            last_val = row
        if n % cfg.ckpt_interval == 0 or n == cfg.steps:
            save_checkpoint(eng, opt, n, run_dir / f"ckpt-{n}.pt")
    metrics.close()
    if cfg.val_gen_samples and answer_val and last_val:
        _write_gates(run_dir, last_val, load_control_result(cfg.text_qa_result))
    return run_dir


def _squad_paragraph_examples(examples: list[dict], eos_id: int) -> list[dict]:
    """Whole-paragraph <full> examples, one per distinct SQuAD paragraph (examples repeat a
    paragraph once per question)."""
    seen, out = set(), []
    for ex in examples:
        key = tuple(ex["chunks"].flatten().tolist())
        if key not in seen:
            seen.add(key)
            out.append(full_example(ex, eos_id))
    return out


def _write_gates(run_dir: Path, row: dict, control: dict | None) -> None:
    g = stage3_gates(row, control)
    d = run_dir / "eval"
    d.mkdir(exist_ok=True)
    (d / "stage3.json").write_text(json.dumps(g, indent=2))
    (d / "stage3.md").write_text(stage3_markdown(g) + "\n")


@torch.no_grad()
def _full_gap(eng, chunks: torch.Tensor, batch_size: int) -> float:
    """Mean <full> val loss with zeroed memory minus with real memory (nats)."""
    eng.eval()
    gap = 0.0
    for i in range(0, len(chunks), batch_size):
        b = chunks[i : i + batch_size].to(eng.cfg.device)
        mem = eng.encode(b)
        ctrl = eng._control("full", len(b))
        eos = torch.full((len(b), 1), eng.eos_id, dtype=torch.long, device=b.device)
        tgt = torch.cat([b, eos], dim=1)
        real = eng.decoder.loss(mem, ctrl, tgt)
        zero = eng.decoder.loss(torch.zeros_like(mem), ctrl, tgt)
        gap += float(zero - real) * len(b)
    eng.train()
    return gap / len(chunks)


def _generation_val(eng, cfg: Config, val_chunks, terse_val, answer_val) -> dict:
    """QA F1 under real/shuffled/zeroed memory, <full> gap and density metrics (Stage 3 gates)."""
    n = cfg.val_gen_samples
    qa = evaluate_qa(eng, answer_val[:n], zero_shot=False, seed=cfg.qa_eval_seed)
    row = {f"qa_{c}_{m}": qa[c][m] for c in ("real", "shuffled", "zeroed") for m in ("em", "f1")}
    row["full_gap"] = _full_gap(eng, val_chunks, cfg.batch_size)
    if terse_val:
        d = evaluate_density(eng, terse_val[:n], max_full_tokens=cfg.val_max_full_tokens or None)
        row.update({k: d[k] for k in ("length_ratio", "terse_article_ratio", "full_article_ratio",
                                      "terse_rouge_l", "baseline_rouge_l")})
    eng.train()
    return row


@torch.no_grad()
def _terse_val_loss(eng, examples: list[dict], batch_size: int) -> float:
    eng.eval()
    total = 0.0
    for i in range(0, len(examples), batch_size):
        b = examples[i : i + batch_size]
        total += float(eng.terse_loss(b)) * len(b)
    eng.train()
    return total / len(examples)
