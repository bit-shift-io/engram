"""Control tokens for the decoder."""

CONTROL_TOKENS = ["<full>", "<terse>"]


def add_control_tokens(tokenizer, model) -> dict[str, int]:
    """Register control tokens with the tokenizer and resize the model's embeddings.

    Returns {token: id}. The resized base rows stay frozen; trainable control
    embeddings live in engram.model.memory.ControlEmbedding.
    """
    tokenizer.add_special_tokens({"additional_special_tokens": CONTROL_TOKENS})
    model.resize_token_embeddings(len(tokenizer))
    for p in model.parameters():
        p.requires_grad_(False)
    return {t: tokenizer.convert_tokens_to_ids(t) for t in CONTROL_TOKENS}
