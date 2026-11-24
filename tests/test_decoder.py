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
