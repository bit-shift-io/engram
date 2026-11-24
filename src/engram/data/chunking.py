"""Split token streams into fixed-size chunks."""


def chunk_tokens(ids: list[int], size: int) -> list[list[int]]:
    """Whole chunks of exactly `size` tokens; the short tail is dropped."""
    return [ids[i : i + size] for i in range(0, len(ids) - size + 1, size)]
