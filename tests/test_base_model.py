import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from engram.config import Config
from engram.model.base import freeze, generate, load_base_model


def tiny_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    )
    return LlamaForCausalLM(cfg)


def test_freeze_makes_all_params_untrainable():
    model = freeze(tiny_model())
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.training


def test_generate_returns_new_tokens_on_configured_device():
    model = freeze(tiny_model())
    out = generate(model, torch.tensor([[1, 2, 3]]), device="cpu", max_new_tokens=4)
    assert out.shape[1] >= 1
    assert out.device.type == "cpu"


@pytest.mark.slow
def test_real_base_model_loads_frozen_and_generates():
    model, tok = load_base_model(Config())
    assert not any(p.requires_grad for p in model.parameters())
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    out = generate(model, ids, device="cpu", max_new_tokens=5)
    assert len(tok.decode(out[0]).strip()) > 0
