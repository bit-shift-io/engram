"""Regression: the decoder must reconstruct from *memory*, not memorise chunks in its own LoRA.

Overfitting one chunk can't tell these apart; with several distinct chunks only a decoder that
reads its memory can tell which chunk to emit, so real memory must beat shuffled memory.
"""

import torch


def teacher_forced_accuracy(eng, memory, chunks) -> float:
    """Argmax accuracy on the chunk tokens given memory (the tiny model's frozen LM head caps
    logit size, so its loss can't fall far; argmax accuracy still shows what it decodes)."""
    ctrl = eng._control("full", len(chunks))
    prefix = eng.decoder._prefix(memory, ctrl)
    x = torch.cat([prefix, eng.lm.get_input_embeddings()(chunks)], dim=1)
    with torch.no_grad(), eng.decoder._ctx():
        logits = eng.lm(inputs_embeds=x).logits[:, prefix.shape[1] - 1 : -1]
    return float((logits.argmax(-1) == chunks).float().mean())


def _accuracies(eng, chunks):
    with torch.no_grad():
        mem = eng.encode(chunks)
    return (teacher_forced_accuracy(eng, mem, chunks),
            teacher_forced_accuracy(eng, mem.roll(1, 0), chunks))


def test_reconstruction_uses_memory_across_distinct_chunks(tiny_engram):
    # Decoder LoRA on: a random frozen tiny decoder can't learn to read memory by itself.
    torch.manual_seed(0)
    eng = tiny_engram(chunk_size=16, k=4)
    chunks = torch.randint(2, 64, (8, 16))
    params = eng.trainable_parameters()
    opt = torch.optim.Adam(params, lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        eng.loss(chunks).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    real, shuf = _accuracies(eng, chunks)
    assert real >= 0.9 and shuf <= 0.6, (real, shuf)


def test_frozen_decoder_has_no_decoder_lora(tiny_engram):
    eng = tiny_engram(chunk_size=8, k=4, decoder_lora=False)
    names = [n for n, p in eng.named_parameters() if p.requires_grad]
    assert any("lora_" in n and ".encoder." in n for n in names)
    assert not any("lora_" in n and ".decoder." in n for n in names)
    eng.loss(torch.randint(2, 64, (2, 8))).backward()
    enc = [p for n, p in eng.named_parameters() if "lora_B" in n and ".encoder." in n]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc)


def test_segment_residual_makes_memory_chunk_specific_at_init(tiny_engram):
    torch.manual_seed(0)
    eng = tiny_engram(chunk_size=16, k=4, memory_segment_residual=True)
    chunks = torch.randint(2, 64, (2, 16))
    with torch.no_grad():
        mem = eng.encode(chunks)
        tok = eng.lm.get_input_embeddings()(chunks.reshape(-1, 16))
        res = eng.encoder.residual(tok)
    assert mem.shape == (2, 4, eng.hidden_size)
    seg = tok.reshape(2, 4, 4, -1).mean(2)
    cos = torch.nn.functional.cosine_similarity(res, seg, dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-4)
    rms = res.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, eng.encoder.embed_rms), rtol=2e-2)
    # the two chunks' memories differ
    assert (mem[0] - mem[1]).abs().mean() > 0.1 * mem.abs().mean()


def test_memory_layer_mask_dims_and_pooled_residual(tiny_engram):
    torch.manual_seed(0)
    chunks = torch.randint(2, 64, (2, 16))
    eng = tiny_engram(chunk_size=16, k=4, memory_layer=1, memory_mask_dims=(3, 5),
                      memory_pooled_residual=True)
    mem = eng.encode(chunks)
    assert mem.shape == (2, 4, eng.hidden_size)
    mem.sum().backward()
    assert eng.encoder.queries.weight.grad is not None
    # masked dims cannot influence memory: perturbing them in the projection input is a no-op
    enc = eng.encoder
    seen = {}
    enc.proj.register_forward_pre_hook(lambda m, a: seen.setdefault("h", a[0].detach()))
    with torch.no_grad():
        enc(chunks)
    assert torch.all(seen["h"][..., [3, 5]] == 0)
    # pooled residual makes memory chunk-specific even with identical query states
    with torch.no_grad():
        m = eng.encode(chunks)
    assert (m[0] - m[1]).abs().mean() > 0.05 * m.abs().mean()
