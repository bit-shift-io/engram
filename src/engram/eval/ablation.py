"""Memory ablation transforms: real, zeroed, shuffled (from a different chunk)."""

import torch


def zero_memory(memory: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(memory)


def shuffle_memory(memory: torch.Tensor) -> torch.Tensor:
    """Derangement along the batch dim: every row gets another row's memory."""
    if memory.shape[0] < 2:
        raise ValueError("shuffled memory needs at least 2 chunks")
    return torch.roll(memory, shifts=1, dims=0)


CONDITIONS = {"real": lambda m: m, "zeroed": zero_memory, "shuffled": shuffle_memory}
