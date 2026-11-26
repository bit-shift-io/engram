import torch


def test_memory_and_control_positions_contribute_no_loss(tiny_engram):
    eng = tiny_engram(chunk_size=8, k=4)
    memory = torch.randn(1, 4, eng.hidden_size)
    ctrl = torch.tensor([[eng.control_ids["<full>"]]])
    target = torch.randint(2, 64, (1, 5))
    base = eng.decoder.loss(memory, ctrl, target)
    # Reference: cross-entropy over target tokens only (prefix length 4 memory + 1 control).
    logits = eng.lm(inputs_embeds=torch.cat(
        [memory, eng.control(ctrl), eng.lm.get_input_embeddings()(target)], 1)).logits
    expected = torch.nn.functional.cross_entropy(logits[0, 4:-1], target[0])
    assert torch.allclose(base, expected, atol=1e-5)


def test_generate_returns_string(tiny_engram):
    eng = tiny_engram(chunk_size=8, k=4)
    out = eng.generate("w1 w2 w3 w4 w5", density="full", max_new_tokens=5)
    assert isinstance(out, str)
    assert isinstance(eng.generate("w1 w2", density="terse", max_new_tokens=3), str)


def test_fact_weights_mean_one_capped_detached_and_zero_on_pad():
    from engram.model.decoder import fact_weights
    losses = torch.tensor([[0.1, 20.0, 1.0, 1.0, 9.0], [2.0, 2.0, 0.5, 0.0, 0.0]],
                          requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    w = fact_weights(losses * 1.0, mask, cap=2.0)
    assert not w.requires_grad
    assert (w[1, 3:] == 0).all()
    assert (w <= 2.0 + 1e-5).all()
    means = (w * mask).sum(1) / mask.sum(1)
    assert torch.allclose(means, torch.ones(2), atol=1e-4)


def test_unit_weights_match_unweighted_loss(tiny_engram):
    eng = tiny_engram(chunk_size=8, k=4)
    memory = torch.randn(2, 4, eng.hidden_size)
    ctrl = torch.full((2, 1), eng.control_ids["<full>"])
    target = torch.randint(2, 64, (2, 6))
    base = eng.decoder.loss(memory, ctrl, target)
    assert torch.allclose(eng.decoder.loss(memory, ctrl, target, weights=torch.ones(2, 6)),
                          base, atol=1e-5)
    assert torch.allclose(eng.decoder.token_losses(memory, ctrl, target).mean(), base, atol=1e-5)


def test_token_predicted_without_memory_gets_lower_weight(tiny_engram):
    from engram.model.decoder import fact_weights
    eng = tiny_engram(chunk_size=8, k=4)
    zero = torch.zeros(1, 4, eng.hidden_size)
    ctrl = torch.tensor([[eng.control_ids["<full>"]]])
    emb = eng.lm.get_input_embeddings()
    # First target token: what the zeroed-memory decoder predicts; second: its least likely token.
    logits = eng.lm(inputs_embeds=torch.cat([zero, eng.control(ctrl)], 1)).logits
    a = logits[0, -1].argmax().view(1, 1)
    logits = eng.lm(inputs_embeds=torch.cat([zero, eng.control(ctrl), emb(a)], 1)).logits
    z = logits[0, -1].argmin().view(1, 1)
    target = torch.cat([a, z], 1)
    with torch.no_grad():
        ref = eng.decoder.token_losses(zero, ctrl, target)
    w = fact_weights(ref, torch.ones_like(target), cap=5.0)
    assert w[0, 0] < w[0, 1]
