import torch


def test_encode_n_chunks_gives_n_times_k_memory_vectors(tiny_engram):
    eng = tiny_engram(chunk_size=8, k=4)
    ids = torch.randint(2, 64, (2, 3 * 8))
    mem = eng.encode(ids)
    assert mem.shape == (2, 3 * 4, eng.hidden_size)


def test_only_lora_queries_projection_and_control_rows_are_trainable(tiny_engram):
    eng = tiny_engram()
    names = [n for n, p in eng.named_parameters() if p.requires_grad]
    allowed = ("lora_", "queries.", "proj.", "control.")
    assert names and all(any(a in n for a in allowed) for n in names)
    assert any("lora_" in n and ".encoder." in n for n in names)
    assert any("lora_" in n and ".decoder." in n for n in names)
    assert not eng.lm.get_input_embeddings().weight.requires_grad


def test_backward_gives_finite_loss_and_grads_reach_both_adapters(tiny_engram):
    eng = tiny_engram(chunk_size=8, k=4)
    loss = eng.loss(torch.randint(2, 64, (2, 16)))
    assert torch.isfinite(loss)
    loss.backward()
    grads = {n: p.grad for n, p in eng.named_parameters() if p.requires_grad}
    for key in (".encoder.", ".decoder.", "queries.", "proj.", "control."):
        assert any(key in n and g is not None and g.abs().sum() > 0 for n, g in grads.items()), key
    assert all(p.grad is None for p in eng.parameters() if not p.requires_grad)
