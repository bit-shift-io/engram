"""Frozen base model loading and greedy generation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from engram.config import Config


def freeze(model: PreTrainedModel) -> PreTrainedModel:
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def load_base_model(cfg: Config):
    """Load the configured base model, frozen, on cfg.device. Returns (model, tokenizer)."""
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
    return freeze(model), tok


@torch.no_grad()
def generate(model, input_ids: torch.Tensor, device: str, max_new_tokens: int = 20) -> torch.Tensor:
    """Greedy-decode; returns only the newly generated token ids."""
    input_ids = input_ids.to(device)
    out = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=model.config.eos_token_id or 0,
    )
    return out[:, input_ids.shape[1]:]


ADAPTERS = ("encoder", "decoder")


def add_lora(model: PreTrainedModel, r: int = 8, alpha: int = 16,
             targets=("q_proj", "k_proj", "v_proj", "o_proj"), decoder: bool = True):
    """Wrap a frozen model with named LoRA adapters: `encoder`, plus `decoder` if requested."""
    from peft import LoraConfig, get_peft_model

    def lora_cfg():
        return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=list(targets))

    peft_model = get_peft_model(model, lora_cfg(), adapter_name=ADAPTERS[0])
    if decoder:
        peft_model.add_adapter(ADAPTERS[1], lora_cfg())
    use_adapter(peft_model, ADAPTERS[0])
    return peft_model


def use_adapter(peft_model, name: str) -> None:
    """Activate one adapter; keep every adapter trainable so gradients reach both."""
    peft_model.set_adapter(name)
    for n, p in peft_model.named_parameters():
        p.requires_grad_("lora_" in n)
