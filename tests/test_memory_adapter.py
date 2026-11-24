"""Decoder memory adapter: k/v deltas at memory positions only."""

import torch

from engram.train.trainer import load_checkpoint, save_checkpoint


def _randomise(eng):
    with torch.no_grad():
        for p in eng.decoder.mem_adapter.B:
            p.normal_(0, 0.5)


def test_adapter_is_noop_without_memory_prefix(tiny_engram):
    torch.manual_seed(0)
    eng = tiny_engram(chunk_size=8, k=4, decoder_lora=False, decoder_memory_adapter_rank=4)
    _randomise(eng)
    ids = torch.randint(2, 64, (2, 10))
    with eng.decoder._ctx():
        base = eng.lm(input_ids=ids).logits
        with eng.decoder._mem(0):
            again = eng.lm(input_ids=ids).logits
    assert torch.equal(base, again)
    # encoder output is unaffected by the adapter (n_mem is 0 outside decoder calls)
    eng2 = tiny_engram(chunk_size=8, k=4, decoder_lora=False)
    torch.manual_seed(1)
    chunks = torch.randint(2, 64, (2, 8))
    eng2.load_state_dict({k: v for k, v in eng.state_dict().items() if "mem_adapter" not in k},
                         strict=False)
    assert torch.allclose(eng.encode(chunks), eng2.encode(chunks))


def test_adapter_changes_only_memory_kv(tiny_engram):
    torch.manual_seed(0)
    eng = tiny_engram(chunk_size=8, k=4, decoder_lora=False, decoder_memory_adapter_rank=4)
    mem = torch.randn(1, 4, eng.hidden_size)
    ctrl = eng._control("full", 1)
    tgt = torch.randint(2, 64, (1, 6))
    l0 = float(eng.decoder.loss(mem, ctrl, tgt))
    _randomise(eng)
    assert float(eng.decoder.loss(mem, ctrl, tgt)) != l0
    # k_proj/v_proj outputs at non-memory positions are identical to the frozen base
    seen = []
    mods = [m for n, m in eng.lm.named_modules() if n.endswith("self_attn.k_proj")]
    h = mods[0].register_forward_hook(lambda m, a, o: seen.append(o.detach().clone()))
    x = torch.cat([eng.decoder._prefix(mem, ctrl), eng.lm.get_input_embeddings()(tgt)], 1)
    with eng.decoder._ctx():
        eng.lm(inputs_embeds=x)
        with eng.decoder._mem(4):
            eng.lm(inputs_embeds=x)
    h.remove()
    # hook order: our probe was registered after the adapter, so it sees the adapted output
    assert torch.equal(seen[0][:, 4:], seen[1][:, 4:])
    assert not torch.equal(seen[0][:, :4], seen[1][:, :4])


def test_adapter_gets_grads_and_generate_runs(tiny_engram):
    torch.manual_seed(0)
    eng = tiny_engram(chunk_size=8, k=4, decoder_lora=False, decoder_memory_adapter_rank=4)
    eng.loss(torch.randint(2, 64, (2, 8))).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in eng.decoder.mem_adapter.B)
    _randomise(eng)
    # KV-cache greedy decode equals a full re-run decode for the first tokens
    mem = eng.encode(torch.randint(2, 64, (1, 8)))
    ctrl = eng._control("full", 1)
    out = eng.decoder.generate(mem, ctrl, 3, eos_id=-1)
    x = eng.decoder._prefix(mem, ctrl)
    ref = []
    with torch.no_grad():
        for _ in range(3):
            with eng.decoder._ctx(), eng.decoder._mem(4):
                nxt = int(eng.lm(inputs_embeds=x).logits[0, -1].argmax())
            ref.append(nxt)
            x = torch.cat([x, eng.lm.get_input_embeddings()(torch.tensor([[nxt]]))], 1)
    assert out == ref


def test_adapter_checkpoint_roundtrip(tiny_engram, tmp_path):
    eng = tiny_engram(chunk_size=8, k=4, decoder_lora=False, decoder_memory_adapter_rank=4)
    _randomise(eng)
    opt = torch.optim.Adam(eng.trainable_parameters())
    save_checkpoint(eng, opt, 1, tmp_path / "c.pt")
    eng2 = tiny_engram(chunk_size=8, k=4, decoder_lora=False, decoder_memory_adapter_rank=4)
    load_checkpoint(eng2, tmp_path / "c.pt")
    for a, b in zip(eng.decoder.mem_adapter.B, eng2.decoder.mem_adapter.B):
        assert torch.equal(a, b)
