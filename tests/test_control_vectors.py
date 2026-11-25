"""Control vectors: a learned per-flag vector added to every decoder layer's residual stream at
control + target positions, so the density flag can't be diluted by distance."""

import torch

from engram.train.trainer import load_checkpoint, save_checkpoint


def _randomise(eng):
    with torch.no_grad():
        eng.decoder.ctrl_vec.vec.normal_(0, 0.5)


def _layer_outputs(eng, x, n_mem, flags):
    seen = []
    layers = eng.decoder.ctrl_vec.layers
    hs = [layers[0].register_forward_hook(lambda m, a, o: seen.append(o[0].detach().clone()))]
    with eng.decoder._ctx(), eng.decoder._flags(flags, n_mem):
        eng.lm(inputs_embeds=x)
    for h in hs:
        h.remove()
    return seen[0]


def test_control_vectors_are_noop_outside_decoder_calls(tiny_engram):
    torch.manual_seed(0)
    eng = tiny_engram(chunk_size=8, k=4, decoder_lora=False, control_vectors=True)
    chunks = torch.randint(2, 64, (2, 8))
    before = eng.encode(chunks)
    ids = torch.randint(2, 64, (2, 10))
    with eng.decoder._ctx():
        base = eng.lm(input_ids=ids).logits
    _randomise(eng)
    assert torch.allclose(eng.encode(chunks), before)
    with eng.decoder._ctx():
        assert torch.equal(eng.lm(input_ids=ids).logits, base)


def test_control_vectors_leave_memory_positions_and_separate_full_from_terse(tiny_engram):
    torch.manual_seed(0)
    eng = tiny_engram(chunk_size=8, k=4, decoder_lora=False, control_vectors=True)
    _randomise(eng)
    mem = torch.randn(1, 4, eng.hidden_size)
    tgt = torch.randint(2, 64, (1, 6))
    ctrl = eng._control("full", 1)
    x = torch.cat([eng.decoder._prefix(mem, ctrl), eng.lm.get_input_embeddings()(tgt)], 1)
    full_id = eng.decoder.ctrl_vec.flag_index(ctrl[:, 0])
    terse_id = eng.decoder.ctrl_vec.flag_index(eng._control("terse", 1)[:, 0])
    off = _layer_outputs(eng, x, 4, None)
    full = _layer_outputs(eng, x, 4, full_id)
    terse = _layer_outputs(eng, x, 4, terse_id)
    assert torch.equal(off[:, :4], full[:, :4])  # memory positions untouched
    assert not torch.allclose(off[:, 4:], full[:, 4:])  # control + target positions shifted
    assert not torch.allclose(full[:, 4:], terse[:, 4:])  # flags differ from each other


def test_control_vectors_get_grads_and_cached_generate_matches_full_rerun(tiny_engram):
    torch.manual_seed(0)
    eng = tiny_engram(chunk_size=8, k=4, decoder_lora=False, control_vectors=True)
    eng.loss(torch.randint(2, 64, (2, 8))).backward()
    assert eng.decoder.ctrl_vec.vec.grad is not None
    assert eng.decoder.ctrl_vec.vec.grad.abs().sum() > 0
    _randomise(eng)
    mem = eng.encode(torch.randint(2, 64, (1, 8)))
    ctrl = eng._control("terse", 1)
    flags = eng.decoder.ctrl_vec.flag_index(ctrl[:, 0])
    out = eng.decoder.generate(mem, ctrl, 3, eos_id=-1)
    x = eng.decoder._prefix(mem, ctrl)
    ref = []
    with torch.no_grad():
        for _ in range(3):
            with eng.decoder._ctx(), eng.decoder._flags(flags, 4):
                nxt = int(eng.lm(inputs_embeds=x).logits[0, -1].argmax())
            ref.append(nxt)
            x = torch.cat([x, eng.lm.get_input_embeddings()(torch.tensor([[nxt]]))], 1)
    assert out == ref


def test_control_vectors_checkpoint_roundtrip(tiny_engram, tmp_path):
    eng = tiny_engram(chunk_size=8, k=4, decoder_lora=False, control_vectors=True)
    _randomise(eng)
    save_checkpoint(eng, torch.optim.Adam(eng.trainable_parameters()), 1, tmp_path / "c.pt")
    eng2 = tiny_engram(chunk_size=8, k=4, decoder_lora=False, control_vectors=True)
    load_checkpoint(eng2, tmp_path / "c.pt")
    assert torch.equal(eng.decoder.ctrl_vec.vec, eng2.decoder.ctrl_vec.vec)
