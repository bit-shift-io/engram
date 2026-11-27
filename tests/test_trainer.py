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


def test_stage3_trains_answer_full_terse_mix_from_stage2_shaped_checkpoint(tmp_path):
    from engram.data.compression import make_example
    from engram.data.qa import make_qa_example

    tr, va = data()
    tok = make_tiny_tokenizer()
    terse = [make_example(tok, " ".join(f"w{(i + j) % 60}" for j in range(5 + 6 * i)), "w1 w2",
                          8, 4) for i in range(6)]
    qa = [make_qa_example(tok, " ".join(f"w{(i + j) % 60}" for j in range(4 + 7 * i)),
                          f"w{40 + i} w50", f"w{i + 2}", 8, 4) for i in range(4)]
    cfg1 = cfg_for(tmp_path, control_vectors=True, full_ratio=0.5, multi_full_ratio=0.5)
    ck = train(build(cfg1), cfg1, tr, va, terse_train=terse) / "ckpt-5.pt"
    saved = torch.load(ck)
    for n in ("control.weight", "decoder.ctrl_vec.vec"):
        saved["params"][n] = saved["params"][n][:2].clone()  # Stage 2 had <full>, <terse> only
    torch.save(saved, ck)

    cfg = cfg_for(tmp_path / "s3", init_from=str(ck), control_vectors=True, answer_ratio=0.5,
                  full_ratio=0.5, multi_full_ratio=0.5, encoder_lr_scale=0.33, steps=8)
    run = train(build(cfg), cfg, tr, va, terse_train=terse, terse_val=terse[:2],
                answer_train=qa, answer_val=qa[:2])
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    steps = [r for r in rows if "train_loss" in r]
    assert all(math.isfinite(r["train_loss"]) for r in steps)
    assert sum(r["n_answer"] for r in steps) > 0
    assert sum(r["n_terse"] + r["n_full"] + r["n_multi"] for r in steps) > 0
    assert any(math.isfinite(r.get("answer_val_loss", math.nan)) for r in rows)


STAGE2_LIKE = {"full_gap": 1.2, "terse_article_ratio": 0.10, "full_article_ratio": 0.90,
               "terse_rouge_l": 0.19, "qa_real_f1": 0.30}


def test_stage3_gates_pass_when_near_stage2_and_ratio_high():
    from engram.eval.report import stage3_gates
    g = stage3_gates(STAGE2_LIKE, control={"em": 0.4, "f1": 0.5})
    assert g["regressions"] == [] and g["memory_over_control_f1"] == 0.6
    assert g["recommend_k32"] is False


def test_stage3_gates_flag_each_regression_and_recommend_k32_below_half():
    from engram.eval.report import stage3_gates, stage3_markdown
    row = {"full_gap": 0.6, "terse_article_ratio": 0.25, "full_article_ratio": 0.5,
           "terse_rouge_l": 0.12, "qa_real_f1": 0.2}
    g = stage3_gates(row, control={"em": 0.4, "f1": 0.5})
    assert set(g["regressions"]) == {"full_gap", "terse_article_ratio", "full_article_ratio",
                                     "terse_rouge_l"}
    assert g["recommend_k32"] is True
    md = stage3_markdown(g)
    assert "K=32" in md and "full_gap" in md


def test_stage3_gates_without_control_make_no_k32_call():
    from engram.eval.report import stage3_gates
    g = stage3_gates(STAGE2_LIKE, control=None)
    assert g["memory_over_control_f1"] is None and g["recommend_k32"] is None


def test_stage3_validation_logs_qa_f1_full_gap_density_and_writes_gates(tmp_path):
    from engram.data.compression import make_example
    from engram.data.qa import make_qa_example

    tr, va = data()
    tok = make_tiny_tokenizer()
    terse = [make_example(tok, " ".join(f"w{(i + j) % 60}" for j in range(5 + 6 * i)), "w1 w2",
                          8, 4) for i in range(4)]
    qa = [make_qa_example(tok, " ".join(f"w{(i + j) % 60}" for j in range(4 + 7 * i)),
                          f"w{40 + i} w50", f"w{i + 2}", 8, 4, answers=[f"w{i + 2}"])
          for i in range(4)]
    cfg = cfg_for(tmp_path, control_vectors=True, answer_ratio=0.5, full_ratio=0.5,
                  multi_full_ratio=0.5, steps=2, val_interval=2, val_gen_samples=2,
                  val_max_full_tokens=8)
    run = train(build(cfg), cfg, tr, va, terse_train=terse, terse_val=terse[:3],
                answer_train=qa, answer_val=qa[:3])
    val = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()
           if "val_loss" in l][-1]
    for k in ("qa_real_f1", "qa_shuffled_f1", "qa_zeroed_f1", "full_gap",
              "terse_article_ratio", "full_article_ratio", "terse_rouge_l", "length_ratio"):
        assert math.isfinite(val[k]), k
    gates = json.loads((run / "eval" / "stage3.json").read_text())
    assert "regressions" in gates and (run / "eval" / "stage3.md").exists()


def test_validation_logs_answer_recall_when_enabled(tmp_path):
    from engram.data.qa import make_qa_example

    tr, va = data()
    tok = make_tiny_tokenizer()
    qa = [make_qa_example(tok, " ".join(f"w{(i + j) % 60}" for j in range(4 + 3 * i)),
                          f"w{40 + i}", f"w{i + 2}", 8, 4, answers=[f"w{i + 2}"]) for i in range(3)]
    cfg = cfg_for(tmp_path, steps=2, val_interval=2, val_recall_samples=2)
    run = train(build(cfg), cfg, tr, va, answer_val=qa)
    val = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()
           if "val_loss" in l][-1]
    assert 0.0 <= val["answer_recall"] <= 1.0 and math.isfinite(val["recon_token_f1"])


def test_mixed_sampler_draws_squad_paragraph_full_at_configured_share():
    from engram.data.compression import make_example
    from engram.data.mixed import MixedSampler
    from engram.data.qa import make_qa_example

    tok = make_tiny_tokenizer()
    full = torch.randint(2, 64, (20, 8))
    terse = [make_example(tok, "w1 w2", "w3", 8, 4) for _ in range(10)]
    qa = [make_qa_example(tok, "w1 w2", "w3", "w4", 8, 4) for _ in range(10)]
    para = [{"chunks": q["chunks"], "target": torch.tensor([5, 6, 0]), "density": "full"}
            for q in qa]
    s = MixedSampler(full, terse, full_ratio=0.35 / 0.6, batch_size=3,
                     generator=torch.Generator().manual_seed(0), answer=qa, answer_ratio=0.4,
                     squad_full=para, squad_ratio=0.5)
    names = [k for k, b in (s.next() for _ in range(4000))]
    assert abs(names.count("answer") / 4000 - 0.40) < 0.03
    assert abs(names.count("squad_full") / 4000 - 0.175) < 0.03
    assert abs(names.count("full") / 4000 - 0.175) < 0.03
    assert abs(names.count("terse") / 4000 - 0.25) < 0.03


def test_stage3_gates_flag_collapsed_squad_full_output():
    from engram.eval.report import stage3_gates, stage3_markdown
    ok = {**STAGE2_LIKE, "squad_full_length_ratio": 0.95, "answer_recall": 0.4}
    assert stage3_gates(ok, control=None)["regressions"] == []
    short = {**ok, "squad_full_length_ratio": 0.2, "answer_recall": 0.02}
    g = stage3_gates(short, control=None)
    assert set(g["regressions"]) == {"squad_full_length_ratio", "answer_recall"}
    assert "squad_full_length_ratio" in stage3_markdown(g)


def test_squad_gates_accept_k16_length_but_flag_recall_lost_after_gate():
    from engram.eval.report import stage3_gates
    # K=16 Stage 2's SQuAD <full> length ratio (0.50) is healthy, not a collapse.
    healthy_short = {**STAGE2_LIKE, "squad_full_length_ratio": 0.5, "answer_recall": 0.25}
    assert stage3_gates(healthy_short, control=None)["regressions"] == []
    # Recall that fell well below the 0.25 recall gate means the joint stage lost answers.
    lost = {**healthy_short, "answer_recall": 0.15}
    assert stage3_gates(lost, control=None)["regressions"] == ["answer_recall"]


def test_joint_stage_trains_squad_full_and_gates_include_squad_metrics(tmp_path):
    from engram.data.compression import make_example
    from engram.data.qa import make_qa_example

    tr, va = data()
    tok = make_tiny_tokenizer()
    terse = [make_example(tok, " ".join(f"w{(i + j) % 60}" for j in range(5 + 6 * i)), "w1 w2",
                          8, 4) for i in range(4)]
    qa = [make_qa_example(tok, " ".join(f"w{(i + j) % 60}" for j in range(4 + 7 * i)),
                          f"w{40 + i} w50", f"w{i + 2}", 8, 4, answers=[f"w{i + 2}"])
          for i in range(4)]
    cfg = cfg_for(tmp_path, control_vectors=True, answer_ratio=0.4, full_ratio=0.6,
                  squad_full_ratio=1.0, steps=6, val_interval=6, val_gen_samples=2,
                  val_recall_samples=2, val_max_full_tokens=8)
    run = train(build(cfg), cfg, tr, va, terse_train=terse, terse_val=terse[:3],
                answer_train=qa, answer_val=qa[:3])
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    steps = [r for r in rows if "train_loss" in r]
    assert sum(r["n_squad_full"] for r in steps) > 0 and sum(r["n_full"] for r in steps) == 0
    assert all(math.isfinite(r["train_loss"]) for r in steps)
    gates = json.loads((run / "eval" / "stage3.json").read_text())
    assert gates["metrics"]["squad_full_length_ratio"] is not None
    assert gates["metrics"]["answer_recall"] is not None


def test_fact_weighting_trains_with_finite_loss(tmp_path):
    from engram.train.trainer import fact_weighted_loss
    cfg = cfg_for(tmp_path, fact_weighting=True, decoder_lora=False, decoder_memory_adapter_rank=2)
    eng = build(cfg)
    train_chunks, val_chunks = data()
    loss = fact_weighted_loss(eng, train_chunks[:2])
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in eng.decoder.mem_adapter.B)
    run = train(eng, cfg, train_chunks, val_chunks)
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert all(math.isfinite(r["train_loss"]) for r in rows if "train_loss" in r)


def _squad_cloze_data(n):
    from engram.data.qa import make_qa_example
    tok = make_tiny_tokenizer()
    return [make_qa_example(tok, " ".join(f"w{(i + j) % 60}" for j in range(6)),
                            "w1", f"w{i % 60}", 8, 4, answers=[f"w{i % 60}", f"w{(i + 3) % 60}"])
            for i in range(n)]


def test_squad_cloze_ratio_zero_leaves_batches_unchanged_and_one_swaps_them():
    from engram.data.cloze import ClozeSampler, squad_cloze_examples
    from engram.data.mixed import MixedSampler
    tr, _ = data()
    exs = squad_cloze_examples(make_tiny_tokenizer(), _squad_cloze_data(4), 18)

    def draw(ratio):
        g = torch.Generator().manual_seed(1)
        s = ClozeSampler(MixedSampler(tr, None, 1.0, 2, g), 0.0, g, exs, ratio, 2)
        return [s.next() for _ in range(5)]
    base = []
    g = torch.Generator().manual_seed(1)
    m = MixedSampler(tr, None, 1.0, 2, g)
    base = [m.next() for _ in range(5)]
    assert all(torch.equal(a[1], b[1]) for a, b in zip(draw(0.0), base))
    assert all(d == "squad_cloze" and len(b) == 2 for d, b in draw(1.0))


def test_tiny_run_logs_squad_cloze_val_keys(tmp_path):
    tr, va = data()
    qa = _squad_cloze_data(60)
    cfg = cfg_for(tmp_path, steps=2, val_interval=2, squad_cloze_ratio=0.5, answer_ratio=0.0)
    run = train(build(cfg), cfg, tr, va, answer_train=qa, answer_val=qa)
    val = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()
           if "val_loss" in l][-1]
    for c in ("real", "shuffled", "zeroed"):
        assert math.isfinite(val[f"squad_cloze_val_loss_{c}"])


def test_cloze_and_squad_cloze_ratios_give_overall_shares_of_point_two_and_point_three():
    from collections import Counter

    from engram.data.cloze import ClozeSampler, make_cloze_example, squad_cloze_examples
    from engram.data.mixed import MixedSampler
    tr, _ = data()
    exs = squad_cloze_examples(make_tiny_tokenizer(), _squad_cloze_data(4), 18)
    g = torch.Generator().manual_seed(3)
    s = ClozeSampler(MixedSampler(tr, None, 1.0, 2, g), 0.2, g, exs, 0.375, 2)
    n = 4000
    counts = Counter(s.next()[0] for _ in range(n))
    assert abs(counts["cloze"] / n - 0.2) < 0.03
    assert abs(counts["squad_cloze"] / n - 0.3) < 0.03
    assert abs(counts["full"] / n - 0.5) < 0.03


def test_early_stop_halts_at_step_when_squad_cloze_gap_is_small(tmp_path):
    tr, va = data()
    qa = _squad_cloze_data(60)
    cfg = cfg_for(tmp_path, steps=6, val_interval=2, ckpt_interval=2, squad_cloze_ratio=0.5,
                  early_stop_step=2, early_stop_gap=1e9)
    run = train(build(cfg), cfg, tr, va, answer_train=qa, answer_val=qa)
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert max(r["step"] for r in rows) == 2
    stop = json.loads((run / "eval" / "early_stop.json").read_text())
    assert stop["step"] == 2 and "gap" in stop and stop["threshold"] == 1e9
    assert (run / "ckpt-2.pt").exists()


def test_early_stop_does_not_trigger_when_gap_is_large_enough(tmp_path):
    tr, va = data()
    qa = _squad_cloze_data(60)
    cfg = cfg_for(tmp_path, steps=4, val_interval=2, ckpt_interval=2, squad_cloze_ratio=0.5,
                  early_stop_step=2, early_stop_gap=0.0)
    run = train(build(cfg), cfg, tr, va, answer_train=qa, answer_val=qa)
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert max(r["step"] for r in rows) == 4
    assert not (run / "eval" / "early_stop.json").exists()


def test_squad_cloze_val_max_caps_the_validation_set(tmp_path, monkeypatch):
    import engram.train.trainer as T
    seen = []
    orig = T._terse_val_loss
    monkeypatch.setattr(T, "_terse_val_loss", lambda eng, ex, bs, f=None: (seen.append(len(ex)), orig(eng, ex, bs, f))[1])
    tr, va = data()
    qa = _squad_cloze_data(60)
    cfg = cfg_for(tmp_path, steps=2, val_interval=2, squad_cloze_ratio=0.5, squad_cloze_val_max=4)
    train(build(cfg), cfg, tr, va, answer_train=qa, answer_val=qa)
    assert 4 in seen and max(seen) <= 4 or 4 in seen
