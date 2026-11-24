import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from engram.config import Config
from engram.model.base import freeze

WORDS = ["<eos>", "<unk>"] + [f"w{i}" for i in range(62)]


def make_tiny_tokenizer():
    vocab = {w: i for i, w in enumerate(WORDS)}
    tk = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>", unk_token="<unk>")


def make_tiny_model(hidden=64, layers=2):
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=len(WORDS), hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
        eos_token_id=0,
    )
    return freeze(LlamaForCausalLM(cfg))


@pytest.fixture
def tiny_engram():
    from engram.model.engram import Engram

    def build(chunk_size=8, k=4, **cfg_kw):
        cfg = Config(chunk_size=chunk_size, k=k, **cfg_kw)
        return Engram(make_tiny_model(), make_tiny_tokenizer(), cfg)

    return build
