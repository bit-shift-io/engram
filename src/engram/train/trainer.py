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
from engram.data.cloze import ClozeSampler, make_cloze_example, squad_cloze_examples
from engram.eval.ablation import CONDITIONS, zero_memory
from engram.eval.cloze_recall import select_recall_paragraphs
from engram.eval.density import evaluate_density
from engram.eval.qa import evaluate_qa
from engram.eval.recall import evaluate_recall
from engram.eval.report import load_control_result, stage3_gates, stage3_markdown
from engram.model.decoder import fact_weights
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


def _may_start_fresh(name: str) -> bool:
    """Control vectors and decoder LoRA are zero/no-op at init, so a checkpoint from a stage
    without them (e.g. the SQuAD cloze run) may lack them."""
    return name == "decoder.ctrl_vec.vec" or ("lora_" in name and ".decoder." in name)


def load_checkpoint(eng, path, opt=None) -> int:
    """Load trainable params (and optimizer if given); returns the saved step."""
    ck = torch.load(path)
    named = dict(eng.named_parameters())
    missing = [n for n, p in named.items() if p.requires_grad and n not in ck["params"]
               and not _may_start_fresh(n)]
    if missing:
        raise KeyError(f"checkpoint {path} lacks {len(missing)} params, e.g. {missing[:3]}")
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
    if density == "cloze":
        return eng.terse_loss(cloze_examples(eng, batch))
    if density != "full":  # incl. "squad_cloze": a list of ready cloze examples
        return eng.terse_loss(batch)
    return fact_weighted_loss(eng, batch) if eng.cfg.fact_weighting else reconstruction_loss(eng, batch)


def squad_cloze_val_examples(tokenizer, dev: list[dict], max_len: int) -> list[dict]:
    """Validation cloze examples from SQuAD dev, without the paragraphs the cloze recall eval
    selects from the same `dev` examples (n=40), so that eval stays unseen."""
    held_out = [src for src, _ in select_recall_paragraphs(tokenizer, dev, max_len)]
    return squad_cloze_examples(tokenizer, dev, max_len, exclude=held_out)


def cloze_examples(eng, chunk_ids: torch.Tensor, fallback: bool = True) -> list[dict]:
    """Cloze examples for a [B, C] chunk batch; chunks without fact spans become <full>
    reconstruction examples (fallback) or are dropped. Budget: 2 * (chunk_size + 1) tokens."""
    out = []
    for c in chunk_ids:
        ex = make_cloze_example(eng.tokenizer, c, 2 * (eng.cfg.chunk_size + 1))
        if ex is None and fallback:
            ex = {"chunks": c.reshape(1, -1), "density": "full",
                  "target": torch.cat([c, torch.tensor([eng.eos_id], dtype=c.dtype)])}
        if ex is not None:
            out.append(ex)
    return out


def fact_weighted_loss(eng, chunk_ids: torch.Tensor) -> torch.Tensor:
    """<full> reconstruction loss with each target token weighted by the decoder's loss on it
    under zeroed memory (the memory-necessity eval's zero transform), so tokens the decoder
    already predicts without memory count less (see `fact_weights`)."""
    chunk_ids = chunk_ids.to(eng.cfg.device)
    b = chunk_ids.shape[0]
    memory = eng.encode(chunk_ids)
    eos = torch.full((b, 1), eng.eos_id, dtype=torch.long, device=chunk_ids.device)
    target = torch.cat([chunk_ids, eos], dim=1)
    ctrl = eng._control("full", b)
    with torch.no_grad():
        ref = eng.decoder.token_losses(zero_memory(memory), ctrl, target)
    w = fact_weights(ref, torch.ones_like(target), eng.cfg.fact_weight_cap)
    return eng.decoder.loss(memory, ctrl, target, weights=w)


def train(eng, cfg: Config, train_chunks: torch.Tensor, val_chunks: torch.Tensor,
          terse_train: list[dict] | None = None, terse_val: list[dict] | None = None,
          answer_train: list[dict] | None = None, answer_val: list[dict] | None = None,
          cloze_dev: list[dict] | None = None) -> Path:
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
    cloze_len = 2 * (eng.cfg.chunk_size + 1)
    squad_cloze_train = squad_cloze_val = []
    if cfg.squad_cloze_ratio > 0:
        squad_cloze_train = squad_cloze_examples(eng.tokenizer, answer_train, cloze_len)
        if cloze_dev or answer_val:
            squad_cloze_val = squad_cloze_val_examples(eng.tokenizer, cloze_dev or answer_val, cloze_len)
        if cfg.squad_cloze_val_max:
            squad_cloze_val = squad_cloze_val[: cfg.squad_cloze_val_max]
        if len(squad_cloze_val) % cfg.batch_size == 1:  # a last batch of 1 cannot be shuffled
            squad_cloze_val = squad_cloze_val[:-1]
    sampler = ClozeSampler(sampler, cfg.cloze_ratio, gen, squad_cloze_train,
                           cfg.squad_cloze_ratio, cfg.batch_size)
    cloze_val = cloze_examples(eng, val_chunks, fallback=False) if cfg.cloze_ratio > 0 else []
    if len(cloze_val) % cfg.batch_size == 1:  # a last batch of 1 cannot be shuffled
        cloze_val = cloze_val[:-1]
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
        total, counts = 0.0, {"full": 0, "multi": 0, "squad_full": 0, "terse": 0, "answer": 0, "cloze": 0,
                                  "squad_cloze": 0}
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
        if cfg.cloze_ratio > 0:
            extra["n_cloze"] = counts["cloze"]
        if cfg.squad_cloze_ratio > 0:
            extra["n_squad_cloze"] = counts["squad_cloze"]
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
            if len(cloze_val) >= 2:  # shuffled memory needs another example's memory
                for c, f in CONDITIONS.items():
                    row[f"cloze_val_loss_{c}"] = _terse_val_loss(eng, cloze_val, cfg.batch_size, f)
            if len(squad_cloze_val) >= 2:
                for c, f in CONDITIONS.items():
                    row[f"squad_cloze_val_loss_{c}"] = _terse_val_loss(eng, squad_cloze_val,
                                                                       cfg.batch_size, f)
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
        stop = _early_stop(cfg, n, last_val if n % cfg.val_interval == 0 or n == cfg.steps else None)
        if n % cfg.ckpt_interval == 0 or n == cfg.steps or stop:
            save_checkpoint(eng, opt, n, run_dir / f"ckpt-{n}.pt")
        if stop:
            (run_dir / "eval").mkdir(exist_ok=True)
            (run_dir / "eval" / "early_stop.json").write_text(json.dumps(stop, indent=2))
            break
    metrics.close()
    if cfg.val_gen_samples and answer_val and last_val:
        _write_gates(run_dir, last_val, load_control_result(cfg.text_qa_result))
    return run_dir


def _early_stop(cfg: Config, n: int, row: dict | None) -> dict | None:
    """At step cfg.early_stop_step: stop when SQuAD cloze real and zeroed val losses differ by
    less than cfg.early_stop_gap nats (the memory is not being used for the answers)."""
    if not cfg.early_stop_step or n != cfg.early_stop_step or not row:
        return None
    if "squad_cloze_val_loss_real" not in row:
        return None
    gap = abs(row["squad_cloze_val_loss_zeroed"] - row["squad_cloze_val_loss_real"])
    if gap >= cfg.early_stop_gap:
        return None
    return {"step": n, "gap": gap, "threshold": cfg.early_stop_gap,
            "reason": f"squad cloze real/zeroed val loss gap {gap:.3f} < {cfg.early_stop_gap} at step {n}"}


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
def _terse_val_loss(eng, examples: list[dict], batch_size: int, memory_transform=None) -> float:
    eng.eval()
    total = 0.0
    for i in range(0, len(examples), batch_size):
        b = examples[i : i + batch_size]
        total += float(eng.terse_loss(b, memory_transform)) * len(b)
    eng.train()
    return total / len(examples)
