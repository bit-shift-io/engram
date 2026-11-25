import json
import math
from dataclasses import replace

import torch

from engram.config import Config
from engram.train.trainer import load_checkpoint, lr_at, train
from conftest import make_tiny_model, make_tiny_tokenizer


def cfg_for(tmp_path, **kw):
    base = Config(chunk_size=8, k=4, steps=5, batch_size=2, grad_accum=2, warmup_steps=1,
                  val_interval=5, ckpt_interval=5, run_dir=str(tmp_path), lr=1e-2)
    return replace(base, **kw)


def build(cfg):
    from engram.model.engram import Engram
    return Engram(make_tiny_model(), make_tiny_tokenizer(), cfg)


def data():
    g = torch.Generator().manual_seed(0)
    return torch.randint(2, 64, (16, 8), generator=g), torch.randint(2, 64, (4, 8), generator=g)


def test_lr_schedule_warmup_then_cosine():
    cfg = Config(lr=1.0, steps=10, warmup_steps=2)
    assert lr_at(0, cfg) == 0.5 and lr_at(1, cfg) == 1.0
    assert lr_at(9, cfg) < 0.1


def test_smoke_run_writes_artifacts_and_resumes(tmp_path):
    cfg = cfg_for(tmp_path)
    tr, va = data()
    run = train(build(cfg), cfg, tr, va)
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert sum("train_loss" in r for r in rows) == 5
    assert any("val_loss" in r for r in rows)
    assert (run / "config.yaml").exists()
    ck = run / "ckpt-5.pt"
    assert ck.exists()

    fresh = build(cfg)
    assert load_checkpoint(fresh, ck) == 5
    saved = torch.load(ck)["params"]
    named = dict(fresh.named_parameters())
    assert all(torch.equal(named[n], v) for n, v in saved.items())

    cfg2 = replace(cfg, steps=7, resume=str(ck))
    run2 = train(build(cfg2), cfg2, tr, va)
    assert run2 == run
    steps = [json.loads(l)["step"] for l in (run / "metrics.jsonl").read_text().splitlines()
             if "train_loss" in l]
    assert steps[-2:] == [6, 7]


def test_stage2_runs_from_stage1_checkpoint_with_mixed_batches_and_split_lr(tmp_path):
    from engram.data.compression import make_example
    from engram.train.trainer import param_groups

    cfg1 = cfg_for(tmp_path)
    tr, va = data()
    ck = train(build(cfg1), cfg1, tr, va) / "ckpt-5.pt"

    tok = make_tiny_tokenizer()
    terse = [make_example(tok, " ".join(f"w{(i + j) % 60}" for j in range(5 + 6 * i)), "w1 w2",
                          8, 4) for i in range(6)]
    cfg2 = cfg_for(tmp_path / "s2", init_from=str(ck), encoder_lr_scale=0.1, full_ratio=0.5)
    eng = build(cfg2)
    groups = param_groups(eng, cfg2)
    enc = {id(p) for p in groups[0]["params"]}
    assert all(id(p) in enc for p in eng.encoder.queries.parameters())
    assert all(id(p) in enc for p in eng.encoder.proj.parameters() if p.requires_grad)
    assert all(id(p) not in enc for p in eng.control.parameters())
    lora = {n: id(p) for n, p in eng.named_parameters() if "lora_" in n}
    assert all((v in enc) == (".encoder." in n) for n, v in lora.items())
    assert groups[0]["scale"] == 0.1 and groups[1]["scale"] == 1.0

    run = train(eng, cfg2, tr, va, terse_train=terse)
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    steps = [r for r in rows if "train_loss" in r]
    assert [r["step"] for r in steps] == [1, 2, 3, 4, 5]  # fresh step count
    assert abs(steps[0]["encoder_lr"] - 0.1 * steps[0]["lr"]) < 1e-12
    assert sum(r["n_terse"] for r in steps) > 0 and sum(r["n_full"] for r in steps) > 0


def test_init_from_loads_checkpoint_params(tmp_path):
    cfg1 = cfg_for(tmp_path)
    tr, va = data()
    ck = train(build(cfg1), cfg1, tr, va) / "ckpt-5.pt"
    cfg2 = cfg_for(tmp_path / "s2", init_from=str(ck), steps=1, lr=0.0)
    eng = build(cfg2)
    train(eng, cfg2, tr, va)
    named = dict(eng.named_parameters())
    assert all(torch.equal(named[n], v) for n, v in torch.load(ck)["params"].items())


def test_frozen_decoder_trainer_step_updates_encoder_side_only(tmp_path):
    """decoder_lora=False: the trainer must still update encoder LoRA, queries and projection
    (disable_adapter() in the decoder must not freeze or detach the encoder adapter)."""
    torch.manual_seed(0)
    cfg = cfg_for(tmp_path, steps=2, decoder_lora=False, val_interval=100, ckpt_interval=100)
    eng = build(cfg)
    assert not any(".decoder." in n for n, _ in eng.named_parameters() if "lora_" in n)
    before = {n: p.detach().clone() for n, p in eng.named_parameters() if p.requires_grad}
    tr, va = data()
    train(eng, cfg, tr, va)
    named = dict(eng.named_parameters())
    changed = {n for n, v in before.items() if not torch.equal(v, named[n])}
    assert any("lora_B" in n and ".encoder." in n for n in changed)
    assert any(n.startswith("encoder.queries") for n in changed)
    assert any(n.startswith("encoder.proj") for n in changed)
    assert all(p.requires_grad for n, p in named.items() if "lora_" in n)

def test_stage2_trains_multi_chunk_full_batches_and_logs_their_val_loss(tmp_path):
    from engram.data.compression import make_example

    tr, va = data()
    tok = make_tiny_tokenizer()
    terse = [make_example(tok, " ".join(f"w{(i + j) % 60}" for j in range(5 + 6 * i)), "w1 w2",
                          8, 4) for i in range(6)]
    cfg = cfg_for(tmp_path, full_ratio=0.5, multi_full_ratio=1.0, steps=8)
    run = train(build(cfg), cfg, tr, va, terse_train=terse, terse_val=terse[:2])
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    steps = [r for r in rows if "train_loss" in r]
    assert sum(r["n_multi"] for r in steps) > 0 and sum(r["n_full"] for r in steps) == 0
    assert all(math.isfinite(r["train_loss"]) for r in steps)
    assert any("multi_val_loss" in r for r in rows)
